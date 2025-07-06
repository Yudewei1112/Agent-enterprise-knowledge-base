"""配置管理模块

该模块负责管理应用程序的所有配置信息，包括：
- 环境变量管理
- 模型配置
- API配置
- 系统参数配置
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置管理类"""
    
    def __init__(self):
        """初始化配置"""
        self._model_config = self._load_model_config()
        self._bocha_config = self._load_bocha_config()
        self._system_config = self._load_system_config()
        self._reasoning_config = self._load_reasoning_config()
    
    @staticmethod
    def get_env_or_default(key: str, default: str = "") -> str:
        """获取环境变量，如果不存在则返回默认值
        
        参数:
            key: 环境变量键名
            default: 默认值
            
        返回:
            环境变量值或默认值
        """
        return os.getenv(key, default)
    
    @staticmethod
    def get_env_int(key: str, default: int = 0) -> int:
        """获取环境变量并转换为整数
        
        参数:
            key: 环境变量键名
            default: 默认值
            
        返回:
            环境变量的整数值或默认值
        """
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def get_env_float(key: str, default: float = 0.0) -> float:
        """获取环境变量并转换为浮点数
        
        参数:
            key: 环境变量键名
            default: 默认值
            
        返回:
            环境变量的浮点数值或默认值
        """
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def get_env_bool(key: str, default: bool = False) -> bool:
        """获取环境变量并转换为布尔值
        
        参数:
            key: 环境变量键名
            default: 默认值
            
        返回:
            环境变量的布尔值或默认值
        """
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _load_model_config(self) -> Dict[str, Dict[str, str]]:
        """从环境变量加载模型配置
        
        返回:
            模型配置字典
        """
        return {
            "glm-4-plus": {
                "model": self.get_env_or_default("GLM_4_PLUS_MODEL", "glm-4-plus"),
                "api_key": self.get_env_or_default("GLM_4_PLUS_API_KEY", ""),
                "api_base": self.get_env_or_default("GLM_4_PLUS_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
            },
            "deepseek": {
                "model": self.get_env_or_default("DEEPSEEK_MODEL", "deepseek-reasoner"),
                "api_key": self.get_env_or_default("DEEPSEEK_API_KEY", ""),
                "api_base": self.get_env_or_default("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
            },
            "qwen": {
                "model": self.get_env_or_default("QWEN_MODEL", "qwen3-235b-a22b"),
                "api_key": self.get_env_or_default("QWEN_API_KEY", ""),
                "api_base": self.get_env_or_default("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            },
            "claude3.7": {
                "model": self.get_env_or_default("CLAUDE_MODEL", "claude-3-7-sonnet"),
                "api_key": self.get_env_or_default("CLAUDE_API_KEY", ""),
                "api_base": self.get_env_or_default("CLAUDE_API_BASE", "https://api.anthropic.com/v1")
            }
        }
    
    def _load_bocha_config(self) -> Dict[str, Any]:
        """从环境变量加载Bocha配置
        
        返回:
            Bocha配置字典
        """
        return {
            "api_key": self.get_env_or_default("BOCHA_API_KEY", ""),
            "api_base": self.get_env_or_default("BOCHA_API_BASE", "https://api.bochaai.com/v1/web-search"),
            "timeout": self.get_env_int("BOCHA_TIMEOUT", 30)
        }
    
    def _load_system_config(self) -> Dict[str, Any]:
        """加载系统配置
        
        返回:
            系统配置字典
        """
        return {
            "default_model": self.get_env_or_default("DEFAULT_MODEL", "glm-4-plus"),
            "local_model_path": "local_m3e_model",
            "cache_dir": "cache",
            "upload_dir": "uploads",
            "chunks_dir": "chunks",
            "index_file": "storage/Faiss/faiss_index.faiss",
            "chunks_mapping_file": "storage/Faiss/chunks_mapping.npy",
            "vector_cache_file": "cache/vector_cache.pkl",
            "embedding_cache_file": "cache/embedding_cache.pkl",
            "database_file": "storage/history_messages.db",
            "max_chunk_chars": 500,
            "chunk_overlap": 100,
            "excel_max_chunk_chars": 1000,
            "excel_chunk_overlap": 0,
            "system_check_interval": 60,
            "document_check_interval": 300,  # 文档检查间隔（秒）
            "enable_async_document_check": True,  # 是否启用异步文档检查
            "auto_rebuild_on_change": True,  # 检测到变更时是否自动重建
            "embedding_cache_size": 1000,
            "query_cache_size": 1000,
            "conversation_history_limit": self.get_env_int("CONVERSATION_HISTORY_LIMIT", 10)  # 对话历史消息数量限制
        }
    
    def _load_reasoning_config(self) -> Dict[str, Any]:
        """从环境变量加载ReAct推理配置
        
        返回:
            推理配置字典
        """
        return {
            "max_reasoning_steps": self.get_env_int("MAX_REASONING_STEPS", 5),
            "complexity_threshold": self.get_env_float("COMPLEXITY_THRESHOLD", 0.7),
            "reasoning_temperature": self.get_env_float("REASONING_TEMPERATURE", 0.3),
            "enable_reasoning_logs": self.get_env_bool("ENABLE_REASONING_LOGS", True),
            "enable_complexity_evaluation": self.get_env_bool("ENABLE_COMPLEXITY_EVALUATION", True),
            "max_reasoning_chain_length": self.get_env_int("MAX_REASONING_CHAIN_LENGTH", 20),
            "circuit_breaker_threshold": self.get_env_int("CIRCUIT_BREAKER_THRESHOLD", 3),
            "reasoning_timeout": self.get_env_int("REASONING_TIMEOUT", 300),
            "confidence_threshold": self.get_env_float("CONFIDENCE_THRESHOLD", 0.6),
            "reasoning_logs_dir": "logs/reasoning",
            "complexity_logs_dir": "logs/complexity",
            "reasoning_chains_dir": "reasoning_chains"
        }
    
    @property
    def model_config(self) -> Dict[str, Dict[str, str]]:
        """获取模型配置"""
        return self._model_config
    
    @property
    def bocha_config(self) -> Dict[str, Any]:
        """获取Bocha配置"""
        return self._bocha_config
    
    @property
    def system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self._system_config
    
    @property
    def reasoning_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self._reasoning_config
    
    @property
    def default_model(self) -> str:
        """获取默认模型名称"""
        return self._system_config["default_model"]
    
    def get_model_config(self, model_name: str) -> Dict[str, str]:
        """获取指定模型的配置
        
        参数:
            model_name: 模型名称
            
        返回:
            模型配置字典
            
        异常:
            KeyError: 当模型名称不存在时
        """
        if model_name not in self._model_config:
            raise KeyError(f"不支持的模型名称: {model_name}")
        return self._model_config[model_name]
    
    def is_valid_model(self, model_name: str) -> bool:
        """检查模型名称是否有效
        
        参数:
            model_name: 模型名称
            
        返回:
            是否为有效的模型名称
        """
        return model_name in self._model_config
    
    def get_bocha_config(self) -> Dict[str, Any]:
        """获取Bocha配置
        
        返回:
            Bocha配置字典
        """
        return self._bocha_config
    
    def get_model_call_params(self, model_name: str, base_params: dict = None) -> dict:
        """获取模型特定的调用参数
        
        参数:
            model_name: 模型名称
            base_params: 基础参数字典
            
        返回:
            包含模型特定参数的调用参数字典
        """
        if base_params is None:
            base_params = {}
        
        # 复制基础参数
        call_params = base_params.copy()
        
        # 根据模型名称添加特定参数
        if model_name == "qwen":
            call_params["enable_thinking"] = False  # qwen模型需要禁用thinking模式
        elif model_name == "deepseek":
            call_params["reasoning_effort"] = "medium"  # 设置推理强度
            call_params["temperature"] = 0.3  # 调整温度参数
            call_params["max_tokens"] = 1500  # 增加最大token数
        
        return call_params
    
    def get_reasoning_config(self) -> Dict[str, Any]:
        """获取推理配置
        
        返回:
            推理配置字典
        """
        return self._reasoning_config
    
    def get_max_reasoning_steps(self) -> int:
        """获取最大推理步数"""
        return self._reasoning_config["max_reasoning_steps"]
    
    def get_complexity_threshold(self) -> float:
        """获取复杂度阈值"""
        return self._reasoning_config["complexity_threshold"]
    
    def get_reasoning_temperature(self) -> float:
        """获取推理温度"""
        return self._reasoning_config["reasoning_temperature"]
    
    def is_reasoning_logs_enabled(self) -> bool:
        """检查是否启用推理日志"""
        return self._reasoning_config["enable_reasoning_logs"]
    
    def is_complexity_evaluation_enabled(self) -> bool:
        """检查是否启用复杂度评估"""
        return self._reasoning_config["enable_complexity_evaluation"]
    
    def get_reasoning_logs_dir(self) -> str:
        """获取推理日志目录"""
        return self._reasoning_config["reasoning_logs_dir"]
    
    def get_complexity_logs_dir(self) -> str:
        """获取复杂度日志目录"""
        return self._reasoning_config["complexity_logs_dir"]
    
    def get_reasoning_chains_dir(self) -> str:
        """获取推理链存储目录"""
        return self._reasoning_config["reasoning_chains_dir"]

# 创建全局配置实例
config = Config()