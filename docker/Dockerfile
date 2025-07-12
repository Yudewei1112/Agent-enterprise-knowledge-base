# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY requirements.txt .

# 创建requirements-docker.txt，移除Windows特定依赖并替换为CPU版本的PyTorch
RUN sed '/pywin32/d' requirements.txt | \
    sed 's/torch==2.7.1+cu118/torch==2.7.1+cpu/' | \
    sed 's/torchaudio==2.7.1+cu118/torchaudio==2.7.1+cpu/' | \
    sed 's/torchvision==0.22.1+cu118/torchvision==0.22.1+cpu/' | \
    sed 's/faiss-gpu==1.10.0/faiss-cpu==1.10.0/' > requirements-docker.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-docker.txt

# 创建必要的目录
RUN mkdir -p /app/uploads \
    /app/storage/Faiss \
    /app/storage/graph_rag \
    /app/cache \
    /app/chunks \
    /app/logs/complexity \
    /app/static \
    /app/templates

# 复制应用代码
COPY . .

# 设置权限
RUN chmod -R 755 /app
RUN chmod +x /app/docker/docker-entrypoint.sh
RUN chmod +x /app/docker/health_check.py

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD python /app/docker/health_check.py --simple || exit 1

# 设置启动脚本
ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]

# 启动命令
CMD ["python", "main.py"]