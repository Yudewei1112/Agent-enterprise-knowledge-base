#!/usr/bin/env python3
# setup.py - 项目初始化脚本
import os

def setup_project():
    """初始化项目目录和配置"""
    # 创建必要目录
    dirs = [
        'uploads',           # 文档上传目录
        'cache',             # 缓存目录
        'chunks',            # 文档分块存储
        'storage',           # 数据存储目录
        'storage/Faiss',     # 向量数据库存储
        'logs',              # 日志目录
        'logs/reasoning',    # 推理过程日志
        'logs/complexity',   # 复杂度评估日志
        'reasoning_chains',  # 推理链存储目录
        'static'             # 静态文件目录
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")
    
    # 检查.env文件
    if not os.path.exists('.env'):
        print("⚠️  请复制.env.example为.env并配置API密钥")
        print("   需要配置的环境变量:")
        print("   - OPENAI_API_KEY: OpenAI API密钥")
        print("   - OPENAI_BASE_URL: API基础URL")
        print("   - MAX_REASONING_STEPS: 最大推理步数 (默认: 5)")
        print("   - COMPLEXITY_THRESHOLD: 复杂度阈值 (默认: 0.7)")
        print("   - REASONING_TEMPERATURE: 推理温度 (默认: 0.3)")
    
    # 创建推理配置文件模板
    reasoning_config_template = '''# ReAct推理引擎配置
MAX_REASONING_STEPS=5
COMPLEXITY_THRESHOLD=0.7
REASONING_TEMPERATURE=0.3
ENABLE_REASONING_LOGS=true
ENABLE_COMPLEXITY_EVALUATION=true
'''
    
    if not os.path.exists('reasoning_config.env'):
        with open('reasoning_config.env', 'w', encoding='utf-8') as f:
            f.write(reasoning_config_template)
        print("✅ 创建推理配置模板: reasoning_config.env")
    
    print("🚀 项目初始化完成！")
    print("📋 新增功能说明:")
    print("   - ReAct多跳推理引擎")
    print("   - 智能复杂度判断机制")
    print("   - 推理过程日志记录")
    print("   - 推理链状态管理")

if __name__ == "__main__":
    setup_project()