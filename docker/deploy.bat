@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===========================================
echo 智能多Agent企业知识库系统 - Docker部署
echo ===========================================

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查docker-compose是否安装
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo 错误: docker-compose未安装，请先安装docker-compose
    pause
    exit /b 1
)

REM 检查.env文件
if not exist "../.env" (
    if exist ".env.docker.example" (
        echo 📝 创建.env文件...
        copy ".env.docker.example" "../.env"
        echo ⚠️  请编辑根目录下的.env文件，配置您的API密钥和其他设置
        echo    配置完成后，重新运行此脚本
        pause
        exit /b 1
    ) else (
        echo ❌ 未找到.env.docker.example文件
        pause
        exit /b 1
    )
)

REM 创建必要的目录
echo 📁 创建持久化目录...
if not exist "..\storage" mkdir "..\storage"
if not exist "..\storage\Faiss" mkdir "..\storage\Faiss"
if not exist "..\storage\graph_rag" mkdir "..\storage\graph_rag"
if not exist "..\uploads" mkdir "..\uploads"
if not exist "..\cache" mkdir "..\cache"
if not exist "..\logs" mkdir "..\logs"
if not exist "..\logs\complexity" mkdir "..\logs\complexity"
if not exist "..\local_m3e_model" mkdir "..\local_m3e_model"

REM 构建并启动服务
echo 构建Docker镜像...
docker-compose build
if errorlevel 1 (
    echo 错误: Docker镜像构建失败
    pause
    exit /b 1
)

echo 启动服务...
docker-compose up -d
if errorlevel 1 (
    echo 错误: 服务启动失败
    pause
    exit /b 1
)

REM 等待服务启动
echo 等待服务启动...
timeout /t 10 /nobreak >nul

REM 检查服务状态
echo 检查服务状态...
docker-compose ps

REM 显示日志
echo 显示最近的日志...
docker-compose logs --tail=20

echo.
echo ===========================================
echo 部署完成！
echo ===========================================
echo 服务地址: http://localhost:8000
echo 文档管理: http://localhost:8000/docs_manage
echo MCP管理: http://localhost:8000/mcp.html
echo.
echo 常用命令:
echo   查看日志: docker-compose logs -f
echo   停止服务: docker-compose down
echo   重启服务: docker-compose restart
echo   查看状态: docker-compose ps
echo ===========================================

pause