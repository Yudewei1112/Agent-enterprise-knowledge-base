# Docker 部署文件说明

本目录包含了企业知识库项目的所有 Docker 相关文件，用于容器化部署和管理。

## 📁 文件结构

```
docker/
├── Dockerfile                 # Docker 镜像构建文件
├── docker-compose.yml         # Docker Compose 服务编排文件
├── .dockerignore              # Docker 构建忽略文件
├── .env.docker.example        # Docker 环境变量模板
├── deploy.sh                  # Linux/macOS 一键部署脚本
├── deploy.bat                 # Windows 一键部署脚本
├── docker-entrypoint.sh       # 容器启动入口脚本
├── health_check.py            # 健康检查脚本
├── monitor.py                 # 系统监控脚本
└── README.md                  # 本说明文件
```

## 🚀 快速部署

### 1. 环境准备

确保已安装：
- Docker 20.10+
- Docker Compose 2.0+

### 2. 配置环境变量

```bash
# 复制环境变量模板到项目根目录
cp .env.docker.example ../.env

# 编辑环境变量文件
nano ../.env
```

### 3. 一键部署

**Linux/macOS:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```cmd
deploy.bat
```

## 📋 文件详细说明

### Dockerfile
- 基于 Python 3.11 官方镜像
- 安装系统依赖（tesseract-ocr, poppler-utils 等）
- 处理 Windows 特定依赖的兼容性
- 配置应用运行环境

### docker-compose.yml
- 定义服务配置和端口映射
- 配置环境变量
- 设置数据持久化卷
- 配置网络和健康检查

### 部署脚本
- **deploy.sh/deploy.bat**: 自动化部署流程
- 检查依赖、创建目录、构建镜像、启动服务
- 提供部署状态反馈

### 运行时脚本
- **docker-entrypoint.sh**: 容器启动时的初始化脚本
- **health_check.py**: 应用健康状态检查
- **monitor.py**: 系统性能监控

## 🔧 管理命令

### 服务管理
```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 重建服务
docker-compose up --build -d
```

### 监控和调试
```bash
# 进入容器
docker-compose exec knowledge-base bash

# 健康检查
docker-compose exec knowledge-base python /app/docker/health_check.py

# 系统监控
docker-compose exec knowledge-base python /app/docker/monitor.py

# 查看资源使用
docker stats
```

## 📊 数据持久化

以下目录会自动创建并持久化到主机：
- `../storage/`: 知识库数据存储
- `../uploads/`: 用户上传文件
- `../cache/`: 缓存数据
- `../logs/`: 应用日志
# - `../reasoning_chains/`: 推理链数据  # 已移除推理可视化功能
- `../local_m3e_model/`: 本地模型文件

## 🔍 故障排除

### 常见问题

1. **容器启动失败**
   - 检查 `.env` 文件配置
   - 查看容器日志：`docker-compose logs`

2. **端口冲突**
   - 修改 `docker-compose.yml` 中的端口映射
   - 或停止占用 8000 端口的其他服务

3. **权限问题**
   - 确保数据目录有正确的读写权限
   - Linux/macOS: `chmod -R 755 ../storage ../uploads ../cache ../logs`

4. **API 密钥错误**
   - 检查 `.env` 文件中的 API 密钥配置
   - 确保密钥格式正确且有效

### 日志查看
```bash
# 查看所有日志
docker-compose logs

# 查看特定服务日志
docker-compose logs knowledge-base

# 实时跟踪日志
docker-compose logs -f
```

## 🔄 更新和维护

### 更新应用
```bash
# 拉取最新代码
git pull

# 重建并重启服务
docker-compose up --build -d
```

### 清理资源
```bash
# 清理未使用的镜像
docker image prune

# 清理所有未使用的资源
docker system prune -a
```

### 备份数据
```bash
# 备份所有数据
docker-compose exec knowledge-base tar -czf /app/backup.tar.gz /app/storage /app/uploads

# 复制备份到主机
docker cp $(docker-compose ps -q knowledge-base):/app/backup.tar.gz ./backup.tar.gz
```

## 📞 技术支持

如果遇到问题，请：
1. 查看本文档的故障排除部分
2. 检查容器日志获取详细错误信息
3. 确认环境配置是否正确
4. 提交 Issue 时请附上相关日志信息