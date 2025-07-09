"""L1 Agent RAG模块

这是一个智能RAG代理模块，能够根据查询复杂度自动选择最适合的检索方法：
- 传统RAG：适用于简单的事实查询
- GraphRAG：适用于复杂的关系推理查询

主要组件：
- L1AgentRAG: 核心代理类
- QueryComplexityAnalyzer: 查询复杂度分析器
- L1AgentRAGTool: 工具封装类
- L1AgentConfig: 配置管理

使用示例：
```python
from L1_agent_rag import L1AgentRAG, L1AgentRAGTool

# 创建L1 Agent
agent = L1AgentRAG()
result = await agent.query("什么是人工智能？")

# 使用工具接口
tool = L1AgentRAGTool()
result_json = await tool._arun(query="什么是人工智能？")
```
"""

__version__ = "1.0.0"
__author__ = "L1 Agent Team"
__email__ = "team@l1agent.com"
__description__ = "智能RAG代理，自动选择最适合的检索方法"

# 导入核心组件
from .L1_agent_rag_query_analyzer import QueryComplexityAnalyzer, ComplexityAnalysisResult
from .L1_agent_rag import L1AgentRAG, L1AgentResult
from .L1_agent_rag_tool import L1AgentRAGTool, create_l1_agent_rag_tool, get_l1_agent_rag_tool
from .config import (
    L1AgentConfig, 
    ComplexityAnalysisConfig, 
    GraphRAGConfig, 
    TraditionalRAGConfig,
    get_config,
    update_config,
    load_config_from_dict,
    save_config_to_dict,
    load_config_from_env
)

# 导出的公共接口
from .traditional_rag import TraditionalRAGRetriever

__all__ = [
    'L1AgentRAG',
    'L1AgentResult', 
    'L1AgentRAGTool',
    'QueryComplexityAnalyzer',
    'ComplexityAnalysisResult',
    'TraditionalRAGRetriever',
    'GraphRAGRetriever',
    'GraphRAGBuilder',
    'get_l1_agent_rag_tool',
    'create_l1_agent_rag_tool'
]

# 添加GraphRAG相关导入
from .graphrag import (
    GraphRAGExtractor,
    GraphRAGBuilder, 
    GraphRAGRetriever,
    Entity,
    Relationship,
    Community
)