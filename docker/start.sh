#!/bin/bash

# Docker容器启动脚本
# 用于启动知识库应用

set -e

echo "Starting Knowledge Base Application..."

# 检查必要的目录是否存在
echo "Checking directories..."
mkdir -p /app/storage
mkdir -p /app/uploads
mkdir -p /app/cache
mkdir -p /app/logs

# 设置目录权限
echo "Setting permissions..."
chmod 755 /app/storage
chmod 755 /app/uploads
chmod 755 /app/cache
chmod 755 /app/logs

# 检查环境变量
echo "Checking environment variables..."
if [ -z "$DEFAULT_MODEL" ]; then
    echo "Warning: DEFAULT_MODEL not set, using default value"
    export DEFAULT_MODEL="glm-4-plus"
fi

# 等待一下确保所有服务就绪
echo "Waiting for services to be ready..."
sleep 2

# 启动应用
echo "Starting FastAPI application..."
exec python /app/main.py