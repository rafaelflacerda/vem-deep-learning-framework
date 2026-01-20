"""
Script wrapper para rodar train_gnn.py durante sweeps do WandB.

Converte argumentos de linha de comando que o WandB passa
para o formato --override que train_gnn.py espera.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """
    Lê argumentos da forma --chave=valor e converte para --override.
    
    Exemplo:
    Input:  --model.dropout=0.15 --model.hidden_dim=128
    Output: --override model.dropout=0.15 model.hidden_dim=128
    """
    args = sys.argv[1:]
    
    # Separar argumentos normais de argumentos de sweep
    override_args = []
    normal_args = []
    
    for arg in args:
        if arg.startswith("--") and "=" in arg:
            # Argumento de sweep: --model.dropout=0.15
            key_value = arg[2:]  # Remove --
            override_args.append(key_value)
        else:
            # Argumento normal: --config-path, etc.
            normal_args.append(arg)
    
    # Montar comando final
    cmd = ["python", "scripts/train_gnn.py"]
    
    # Adicionar argumentos normais
    cmd.extend(normal_args)
    
    # Adicionar override se houver argumentos de sweep
    if override_args:
        cmd.append("--override")
        cmd.extend(override_args)
    
    # Executar
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()