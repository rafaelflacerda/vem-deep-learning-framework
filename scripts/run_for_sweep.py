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
    
    # Verificar se blocked está definido
    for arg in args:
        if arg == "--blocked=True" or arg == "--blocked=true":
            print("=" * 70)
            print("CONFIGURAÇÃO DE SWEEP BLOQUEADA")
            print("=" * 70)
            print("Este YAML de sweep está marcado como bloqueado.")
            print("Altere 'blocked.value' para False se quiser usá-lo.")
            print("=" * 70)
            sys.exit(1)
        if arg == "--blocked=False" or arg == "--blocked=false":
            break
    else:
        print("=" * 70)
        print("CONFIGURAÇÃO 'blocked' NÃO DEFINIDA")
        print("=" * 70)
        print("Adicione 'blocked.value: True' ou 'blocked.value: False' no YAML.")
        print("=" * 70)
        sys.exit(1)
    
    # Separar argumentos normais de argumentos de sweep
    override_args = []
    normal_args = []
    
    for arg in args:
        if arg.startswith("--blocked="):
            continue  # Ignora blocked, já foi verificado
        if arg.startswith("--") and "=" in arg:
            key_value = arg[2:]
            override_args.append(key_value)
        else:
            normal_args.append(arg)
    
    # Montar comando final
    cmd = ["python", "scripts/train_gnn.py"]
    cmd.extend(normal_args)
    
    if override_args:
        cmd.append("--override")
        cmd.extend(override_args)
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()