"""L1 Agent RAG配置模块

该模块定义了L1 Agent RAG系统的各种配置参数。
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass

# 加载.env文件
try:
    from dotenv import load_dotenv
    # 加载项目根目录的.env文件
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    # 如果没有安装python-dotenv，尝试手动加载.env文件
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


@dataclass
class ComplexityAnalysisConfig:
    """复杂度分析配置"""
    # 基于规则的分析权重
    rule_based_weight: float = 0.3
    
    # 基于LLM的分析权重
    llm_based_weight: float = 0.7
    
    # 复杂度阈值
    complexity_threshold: float = 0.6
    
    # 复杂查询关键词
    complex_keywords: List[str] = None
    
    # 简单查询关键词
    simple_keywords: List[str] = None
    
    # 关系词汇
    relationship_words: List[str] = None
    
    # 多跳推理词汇
    multi_hop_words: List[str] = None
    
    def __post_init__(self):
        if self.complex_keywords is None:
            self.complex_keywords = [
                '关系', '联系', '影响', '导致', '原因', '结果', '比较', '对比',
                '分析', '评估', '综合', '整体', '全面', '深入', '详细',
                '为什么', '如何', '怎样', '什么时候', '在哪里',
                '关联', '相关', '相互', '之间', '差异', '相同', '不同'
            ]
        
        if self.simple_keywords is None:
            self.simple_keywords = [
                '是什么', '定义', '含义', '概念', '介绍', '简介',
                '时间', '日期', '地点', '位置', '数量', '价格',
                '名称', '姓名', '标题', '题目'
            ]
        
        if self.relationship_words is None:
            self.relationship_words = [
                '关系', '联系', '关联', '相关', '影响', '作用',
                '导致', '引起', '产生', '造成', '带来',
                '依赖', '基于', '来源于', '属于', '包含'
            ]
        
        if self.multi_hop_words is None:
            self.multi_hop_words = [
                '通过', '经过', '然后', '接着', '进而', '从而',
                '最终', '最后', '结果', '导致', '影响',
                '链条', '路径', '过程', '步骤', '阶段'
            ]


@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    # 知识图谱存储路径
    graph_storage_path: str = "storage/graph_rag"
    
    # 知识图谱文件名
    graph_filename: str = "knowledge_graph.json"
    
    # 实体提取配置
    max_entities_per_chunk: int = 20
    
    # 关系提取配置
    max_relationships_per_chunk: int = 15
    
    # 社区检测配置
    community_detection_resolution: float = 1.0
    
    # 摘要生成配置
    max_summary_length: int = 500
    
    # 检索配置
    global_search_top_k: int = 10
    local_search_top_k: int = 5
    
    # 相似度阈值
    similarity_threshold: float = 0.7
    
    # 确保存储目录存在
    def get_graph_path(self) -> str:
        """获取完整的知识图谱路径（兼容性方法，已废弃）"""
        full_path = os.path.join(self.graph_storage_path, self.graph_filename)
        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path
    
    def get_multi_graphs_dir(self) -> str:
        """获取多图谱存储目录路径"""
        multi_graphs_path = os.path.join(self.graph_storage_path, "multi_graphs")
        # 确保目录存在
        os.makedirs(multi_graphs_path, exist_ok=True)
        return multi_graphs_path
    
    def has_existing_graphs(self) -> bool:
        """检查是否存在已构建的知识图谱"""
        multi_graphs_dir = self.get_multi_graphs_dir()
        if not os.path.exists(multi_graphs_dir):
            return False
        
        # 检查目录中是否有.json文件
        import glob
        graph_files = glob.glob(os.path.join(multi_graphs_dir, "*.json"))
        return len(graph_files) > 0


@dataclass
class TraditionalRAGConfig:
    """传统RAG配置"""
    # 默认返回结果数量
    default_top_k: int = 5
    
    # 最大返回结果数量
    max_top_k: int = 20
    
    # 相似度阈值
    similarity_threshold: float = 0.5
    
    # 答案生成配置
    answer_generation_temperature: float = 0.3
    answer_generation_max_tokens: int = 1000


@dataclass
class L1AgentConfig:
    """L1 Agent配置"""
    # 组件配置
    complexity_analysis: ComplexityAnalysisConfig = None
    graph_rag: GraphRAGConfig = None
    traditional_rag: TraditionalRAGConfig = None
    
    # 模型配置
    default_model: str = os.getenv("DEFAULT_MODEL", "glm-4-plus")
    models: Dict[str, Dict[str, Any]] = None
    
    # 性能配置
    enable_caching: bool = True
    cache_ttl: int = 3600  # 缓存时间（秒）
    
    # 日志配置
    enable_detailed_logging: bool = True
    log_level: str = "INFO"
    
    # 错误处理配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 回退策略配置
    enable_fallback: bool = True
    fallback_to_traditional_rag: bool = True
    
    def __post_init__(self):
        if self.complexity_analysis is None:
            self.complexity_analysis = ComplexityAnalysisConfig()
        
        if self.graph_rag is None:
            self.graph_rag = GraphRAGConfig()
        
        if self.traditional_rag is None:
            self.traditional_rag = TraditionalRAGConfig()
        
        if self.models is None:
            self.models = self._load_models_from_env()
        
        # 从.env文件中获取默认模型
        env_default_model = os.getenv("DEFAULT_MODEL")
        if env_default_model:
            self.default_model = env_default_model
    
    def _load_models_from_env(self) -> Dict[str, Dict[str, Any]]:
        """从环境变量中加载模型配置"""
        models = {}
        
        # GLM-4-Plus 配置
        glm_model = os.getenv("GLM_4_PLUS_MODEL")
        glm_api_key = os.getenv("GLM_4_PLUS_API_KEY")
        glm_api_base = os.getenv("GLM_4_PLUS_API_BASE")
        if glm_model and glm_api_key and glm_api_base:
            models[glm_model] = {
                "api_key": glm_api_key,
                "api_base": glm_api_base,
                "model": glm_model,
                "max_tokens": 8000,
                "temperature": 0.3
            }
        
        # DeepSeek 配置
        deepseek_model = os.getenv("DEEPSEEK_MODEL")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
        if deepseek_model and deepseek_api_key and deepseek_api_base:
            models[deepseek_model] = {
                "api_key": deepseek_api_key,
                "api_base": deepseek_api_base,
                "model": deepseek_model,
                "max_tokens": 8000,
                "temperature": 0.3
            }
        
        # Qwen 配置
        qwen_model = os.getenv("QWEN_MODEL")
        qwen_api_key = os.getenv("QWEN_API_KEY")
        qwen_api_base = os.getenv("QWEN_API_BASE")
        if qwen_model and qwen_api_key and qwen_api_base:
            models[qwen_model] = {
                "api_key": qwen_api_key,
                "api_base": qwen_api_base,
                "model": qwen_model,
                "max_tokens": 8000,
                "temperature": 0.3
            }
        
        # Claude 配置
        claude_model = os.getenv("CLAUDE_MODEL")
        claude_api_key = os.getenv("CLAUDE_API_KEY")
        claude_api_base = os.getenv("CLAUDE_API_BASE")
        if claude_model and claude_api_key and claude_api_base and claude_api_key != "your_claude_api_key":
            models[claude_model] = {
                "api_key": claude_api_key,
                "api_base": claude_api_base,
                "model": claude_model,
                "max_tokens": 8000,
                "temperature": 0.3
            }
        
        # 如果没有从环境变量中加载到任何模型，提供默认配置
        if not models:
            default_model_name = os.getenv("DEFAULT_MODEL", "glm-4-plus")
            default_api_base = os.getenv("GLM_4_PLUS_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
            models = {
                default_model_name: {
                    "api_key": "your-glm-api-key",
                    "api_base": default_api_base,
                    "model": default_model_name,
                    "max_tokens": 8000,
                    "temperature": 0.3
                }
            }
        
        return models
    
    def get_model_config(self, model_name: str = None) -> Dict[str, Any]:
        """获取模型配置
        
        参数:
            model_name: 模型名称，如果为None则使用默认模型
            
        返回:
            模型配置字典
        """
        if model_name is None:
            model_name = self.default_model
        
        if model_name in self.models:
            return self.models[model_name]
        else:
            # 返回默认模型配置
            return self.models[self.default_model]
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典
        
        返回:
            配置字典
        """
        return save_config_to_dict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'L1AgentConfig':
        """从字典创建配置
        
        参数:
            config_dict: 配置字典
            
        返回:
            L1Agent配置实例
        """
        return load_config_from_dict(config_dict)


# 默认配置实例
default_config = L1AgentConfig()


# 配置管理函数
def get_config() -> L1AgentConfig:
    """获取配置
    
    返回:
        L1Agent配置实例
    """
    return default_config


def update_config(**kwargs) -> None:
    """更新配置
    
    参数:
        **kwargs: 配置参数
    """
    global default_config
    
    for key, value in kwargs.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
        else:
            print(f"警告: 未知的配置参数 '{key}'")


def load_config_from_dict(config_dict: Dict[str, Any]) -> L1AgentConfig:
    """从字典加载配置
    
    参数:
        config_dict: 配置字典
        
    返回:
        L1Agent配置实例
    """
    # 解析复杂度分析配置
    complexity_config = ComplexityAnalysisConfig()
    if 'complexity_analysis' in config_dict:
        ca_dict = config_dict['complexity_analysis']
        for key, value in ca_dict.items():
            if hasattr(complexity_config, key):
                setattr(complexity_config, key, value)
    
    # 解析GraphRAG配置
    graph_config = GraphRAGConfig()
    if 'graph_rag' in config_dict:
        gr_dict = config_dict['graph_rag']
        for key, value in gr_dict.items():
            if hasattr(graph_config, key):
                setattr(graph_config, key, value)
    
    # 解析传统RAG配置
    traditional_config = TraditionalRAGConfig()
    if 'traditional_rag' in config_dict:
        tr_dict = config_dict['traditional_rag']
        for key, value in tr_dict.items():
            if hasattr(traditional_config, key):
                setattr(traditional_config, key, value)
    
    # 创建主配置
    main_config = L1AgentConfig(
        complexity_analysis=complexity_config,
        graph_rag=graph_config,
        traditional_rag=traditional_config
    )
    
    # 设置其他配置
    for key, value in config_dict.items():
        if key not in ['complexity_analysis', 'graph_rag', 'traditional_rag']:
            if hasattr(main_config, key):
                setattr(main_config, key, value)
    
    return main_config


def save_config_to_dict(config: L1AgentConfig) -> Dict[str, Any]:
    """将配置保存为字典
    
    参数:
        config: L1Agent配置实例
        
    返回:
        配置字典
    """
    return {
        'complexity_analysis': {
            'rule_based_weight': config.complexity_analysis.rule_based_weight,
            'llm_based_weight': config.complexity_analysis.llm_based_weight,
            'complexity_threshold': config.complexity_analysis.complexity_threshold,
            'complex_keywords': config.complexity_analysis.complex_keywords,
            'simple_keywords': config.complexity_analysis.simple_keywords,
            'relationship_words': config.complexity_analysis.relationship_words,
            'multi_hop_words': config.complexity_analysis.multi_hop_words
        },
        'graph_rag': {
            'graph_storage_path': config.graph_rag.graph_storage_path,
            'graph_filename': config.graph_rag.graph_filename,
            'max_entities_per_chunk': config.graph_rag.max_entities_per_chunk,
            'max_relationships_per_chunk': config.graph_rag.max_relationships_per_chunk,
            'community_detection_resolution': config.graph_rag.community_detection_resolution,
            'max_summary_length': config.graph_rag.max_summary_length,
            'global_search_top_k': config.graph_rag.global_search_top_k,
            'local_search_top_k': config.graph_rag.local_search_top_k,
            'similarity_threshold': config.graph_rag.similarity_threshold
        },
        'traditional_rag': {
            'default_top_k': config.traditional_rag.default_top_k,
            'max_top_k': config.traditional_rag.max_top_k,
            'similarity_threshold': config.traditional_rag.similarity_threshold,
            'answer_generation_temperature': config.traditional_rag.answer_generation_temperature,
            'answer_generation_max_tokens': config.traditional_rag.answer_generation_max_tokens
        },
        'default_model': config.default_model,
        'models': config.models,
        'enable_caching': config.enable_caching,
        'cache_ttl': config.cache_ttl,
        'enable_detailed_logging': config.enable_detailed_logging,
        'log_level': config.log_level,
        'max_retries': config.max_retries,
        'retry_delay': config.retry_delay,
        'enable_fallback': config.enable_fallback,
        'fallback_to_traditional_rag': config.fallback_to_traditional_rag
    }


# 环境变量配置
def load_config_from_env() -> L1AgentConfig:
    """从环境变量加载配置
    
    返回:
        L1Agent配置实例
    """
    config = L1AgentConfig()
    
    # 复杂度分析配置
    if os.getenv('L1_COMPLEXITY_THRESHOLD'):
        config.complexity_analysis.complexity_threshold = float(os.getenv('L1_COMPLEXITY_THRESHOLD'))
    
    # GraphRAG配置
    if os.getenv('L1_GRAPH_STORAGE_PATH'):
        config.graph_rag.graph_storage_path = os.getenv('L1_GRAPH_STORAGE_PATH')
    
    # 传统RAG配置
    if os.getenv('L1_DEFAULT_TOP_K'):
        config.traditional_rag.default_top_k = int(os.getenv('L1_DEFAULT_TOP_K'))
    
    # 性能配置
    if os.getenv('L1_ENABLE_CACHING'):
        config.enable_caching = os.getenv('L1_ENABLE_CACHING').lower() == 'true'
    
    if os.getenv('L1_CACHE_TTL'):
        config.cache_ttl = int(os.getenv('L1_CACHE_TTL'))
    
    return config