#!/bin/bash
set -e

echo "🚀 Bootstrap RunPod - VEM TCC"
echo "================================"

REPO_DIR="/workspace/vem-deep-learning-framework"
REPO_URL="https://github.com/rafaelflacerda/vem-deep-learning-framework.git"
BRANCH="runpod-experiments"

# Clone ou atualizar repositório
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "📦 Clonando repositório..."
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "📦 Repositório já existe, atualizando..."
  cd "$REPO_DIR"
  git fetch origin
  cd /workspace
fi

cd "$REPO_DIR"
git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" --track "origin/$BRANCH"
git pull --ff-only 2>/dev/null || echo "⚠️  Não foi possível fazer pull (pode ter mudanças locais)"

# Instalar uv se necessário
if ! command -v uv &> /dev/null; then
  echo "📦 Instalando uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
fi

# Sync dependências
echo "📦 Instalando dependências..."
/root/.local/bin/uv sync --all-extras

# Copiar dados do network volume para o repositório
echo "📊 Copiando dados para o repositório..."
if [ -d "/workspace/data/processed" ]; then
  mkdir -p "$REPO_DIR/data"
  cp -r /workspace/data/processed "$REPO_DIR/data/"
  echo "✅ Dados copiados"
else
  echo "⚠️  Dados não encontrados em /workspace/data/processed/"
fi

echo ""
echo "================================"
echo "✅ SETUP COMPLETO!"
echo "================================"
echo "📁 Repositório: $REPO_DIR"
echo "📊 Dados: $REPO_DIR/data/processed/"
echo "🔧 Ambiente: $REPO_DIR/.venv/"
echo ""
echo "Para começar:"
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "================================"