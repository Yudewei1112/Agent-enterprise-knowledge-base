#!/bin/bash

# Docker部署脚本
# 用于快速部署智能多Agent企业知识库系统

set -e

echo "==========================================="
echo "智能多Agent企业知识库系统 - Docker部署"
echo "==========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查docker-compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 检查.env文件
if [ ! -f "../.env" ]; then
    if [ -f ".env.docker.example" ]; then
        echo "📝 创建.env文件..."
        cp .env.docker.example ../.env
        echo "⚠️  请编辑根目录下的.env文件，配置您的API密钥和其他设置"
        echo "   配置完成后，重新运行此脚本"
        exit 1
    else
        echo "❌ 未找到.env.docker.example文件"
        exit 1
    fi
fi

# 创建必要的目录
echo "📁 创建持久化目录..."
mkdir -p ../storage/Faiss
mkdir -p ../storage/graph_rag
mkdir -p ../uploads
mkdir -p ../cache
mkdir -p ../logs/complexity
mkdir -p ../local_m3e_model

# 设置目录权限
echo "设置目录权限..."
chmod -R 755 ../storage ../uploads ../cache ../logs ../local_m3e_model

# 构建并启动服务
echo "构建Docker镜像..."
docker-compose build

echo "启动服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
echo "检查服务状态..."
docker-compose ps

# 显示日志
echo "显示最近的日志..."
docker-compose logs --tail=20

echo ""
echo "==========================================="
echo "部署完成！"
echo "==========================================="
echo "服务地址: http://localhost:8000"
echo "文档管理: http://localhost:8000/docs_manage"
echo "MCP管理: http://localhost:8000/mcp.html"
echo ""
echo "常用命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  查看状态: docker-compose ps"
echo "==========================================="