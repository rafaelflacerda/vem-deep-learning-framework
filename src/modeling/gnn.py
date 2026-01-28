"""
Graph Neural Network para predição de deslocamentos em vigas 1D.

Arquitetura baseada em MeshGraphNet simplificada:
- Encoder: MLP que projeta node features para espaço latente
- Processor: Camadas de message passing (GATv2 com edge features)
- Decoder: MLP que projeta de volta para predição de deslocamento

MC Dropout é usado para quantificação de incerteza.
LayerNorm é aplicado para estabilidade do treinamento (seguindo MeshGraphNet).
Edge features (distância entre nós) são usadas no message passing.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv  # MUDANÇA: GCNConv -> GATv2Conv


def _get_activation(activation_str: str) -> nn.Module:
    """
    Retorna o módulo de activation baseado em string.

    Suporta: relu, silu, gelu, tanh, selu.
    """
    activation_str = activation_str.lower()

    if activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "silu":
        return nn.SiLU()
    elif activation_str == "gelu":
        return nn.GELU()
    elif activation_str == "tanh":
        return nn.Tanh()
    elif activation_str == "selu":
        return nn.SELU()
    else:
        raise ValueError(f"Activation desconhecida: {activation_str}")


class BeamGNN(nn.Module):
    """
    GNN para predição de deslocamentos em vigas 1D.

    Args:
        input_dim: Dimensão das features de entrada.
        hidden_dim: Dimensão do espaço latente.
        output_dim: Dimensão da saída (1 para deslocamento vertical).
        num_layers: Número de camadas de message passing.
        dropout: Taxa de dropout (usado em treinamento e para MC Dropout na inferência).
        activation: Função de ativação ('relu', 'silu', 'gelu', 'tanh', 'selu').
        use_layer_norm: Se True, aplica LayerNorm após cada camada (exceto saída final).
        edge_dim: Dimensão das edge features (1 para distância entre nós).
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        edge_dim: int = 1,  # NOVO PARÂMETRO
    ):
        super().__init__()

        self.activation = _get_activation(activation)
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        self.edge_dim = edge_dim  # Armazenar para referência

        # Encoder: features brutas -> espaço latente
        # LayerNorm é aplicado após a última Linear, antes da activation
        if use_layer_norm:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
            )

        # Processor: camadas de message passing com GATv2Conv
        # GATv2Conv suporta edge features via parâmetro edge_dim
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_dim=edge_dim,  # NOVO: passa edge_dim para usar edge features
                )
            )

        # LayerNorm para cada camada de convolução
        if use_layer_norm:
            self.conv_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.conv_norms.append(nn.LayerNorm(hidden_dim))

        self.conv_dropout = nn.Dropout(dropout)

        # Decoder: espaço latente -> predição
        # LayerNorm na primeira camada, mas NÃO na saída final
        if use_layer_norm:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,  # NOVO PARÂMETRO
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features de shape (n_nodes, input_dim)
            edge_index: Conectividade de shape (2, n_edges)
            edge_attr: Edge features de shape (n_edges, edge_dim)

        Returns:
            Predições de shape (n_nodes, output_dim)
        """
        # Encoder
        h = self.encoder(x)

        # Processor (message passing com skip connections)
        if self.use_layer_norm:
            for conv, norm in zip(self.convs, self.conv_norms):
                h_new = conv(h, edge_index, edge_attr)  # MUDANÇA: passa edge_attr
                h_new = norm(h_new)
                h_new = self.activation(h_new)
                h_new = self.conv_dropout(h_new)
                h = h + h_new  # Skip connection
        else:
            for conv in self.convs:
                h_new = conv(h, edge_index, edge_attr)  # MUDANÇA: passa edge_attr
                h_new = self.activation(h_new)
                h_new = self.conv_dropout(h_new)
                h = h + h_new  # Skip connection

        # Decoder
        out = self.decoder(h)

        return out

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,  # NOVO PARÂMETRO
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predição com quantificação de incerteza via MC Dropout.

        Executa múltiplos forward passes com dropout ativado e calcula
        média e desvio padrão das predições.

        Args:
            x: Node features.
            edge_index: Conectividade.
            edge_attr: Edge features.
            n_samples: Número de forward passes para Monte Carlo.

        Returns:
            Tupla (mean, std) onde:
                - mean: Média das predições (n_nodes, output_dim)
                - std: Desvio padrão das predições (n_nodes, output_dim)
        """
        self.train()  # Ativa dropout

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, edge_index, edge_attr)  # MUDANÇA: passa edge_attr
                predictions.append(pred)

        predictions = torch.stack(predictions)  # (n_samples, n_nodes, output_dim)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std