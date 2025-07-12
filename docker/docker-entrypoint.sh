#!/bin/bash

# Docker容器启动脚本
# 用于在容器启动时进行初始化检查和配置

set -e

echo "==========================================="
echo "智能多Agent企业知识库系统 - 容器启动"
echo "==========================================="

# 检查环境变量
echo "检查环境变量配置..."
if [ -z "$GLM_4_PLUS_API_KEY" ] && [ -z "$DEEPSEEK_API_KEY" ] && [ -z "$QWEN_API_KEY" ] && [ -z "$CLAUDE_API_KEY" ]; then
    echo "警告: 未检测到任何模型API密钥，请确保至少配置一个模型的API密钥"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p /app/storage/Faiss
mkdir -p /app/storage/graph_rag
mkdir -p /app/uploads
mkdir -p /app/cache
mkdir -p /app/logs/complexity
mkdir -p /app/chunks

# 设置目录权限
echo "设置目录权限..."
chmod -R 755 /app/storage
chmod -R 755 /app/uploads
chmod -R 755 /app/cache
chmod -R 755 /app/logs


# 检查Python依赖
echo "检查Python依赖..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import faiss; print('FAISS已安装')"
python -c "import sentence_transformers; print('SentenceTransformers已安装')"
python -c "import openai; print('OpenAI已安装')"

# 检查CUDA可用性
echo "检查CUDA可用性..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA加速可用"
else
    echo "使用CPU模式"
fi

# 检查本地模型文件
echo "检查本地模型文件..."
if [ -d "/app/local_m3e_model" ]; then
    echo "检测到本地M3E模型文件"
    ls -la /app/local_m3e_model/
else
    echo "未检测到本地M3E模型文件，将从网络下载"
fi

# 检查现有数据
echo "检查现有数据..."
if [ -f "/app/storage/Faiss/faiss_index.faiss" ]; then
    echo "检测到现有FAISS索引文件"
else
    echo "未检测到FAISS索引文件，将在首次运行时创建"
fi

if [ -d "/app/storage/graph_rag" ] && [ "$(ls -A /app/storage/graph_rag)" ]; then
    echo "检测到现有GraphRAG数据"
    echo "GraphRAG文件数量: $(find /app/storage/graph_rag -name '*.json' | wc -l)"
else
    echo "未检测到GraphRAG数据，将在首次运行时创建"
fi

# 检查上传目录
if [ -d "/app/uploads" ] && [ "$(ls -A /app/uploads)" ]; then
    echo "检测到现有文档文件"
    echo "文档数量: $(find /app/uploads -type f | wc -l)"
else
    echo "未检测到文档文件，请通过Web界面上传文档"
fi

# 设置Python路径
export PYTHONPATH=/app:$PYTHONPATH

# 启动健康检查服务（后台）
echo "启动健康检查服务..."
python /app/docker/health_check.py --simple > /dev/null 2>&1 &

echo "==========================================="
echo "初始化完成，启动应用服务..."
echo "==========================================="

# 启动主应用
exec "$@"