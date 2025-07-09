# L1 Agent RAG

一个智能的RAG代理系统，能够根据查询复杂度自动选择最适合的检索方法。

## 🌟 特性

- **智能路由**: 自动分析查询复杂度，选择最适合的检索方法
- **双重检索**: 支持传统RAG和GraphRAG两种检索方式
- **Agent as Tool**: 可以作为工具集成到LangGraph Agent中
- **高度可配置**: 丰富的配置选项，支持自定义参数
- **异步支持**: 完全异步实现，支持高并发
- **详细监控**: 提供详细的执行信息和性能统计

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              L1 Agent RAG Tool                      │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │            L1 Agent RAG                     │    │    │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  │    │    │
│  │  │  │ Query Analyzer  │  │ Method Selector │  │    │    │
│  │  │  └─────────────────┘  └─────────────────┘  │    │    │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  │    │    │
│  │  │  │Traditional RAG  │  │   GraphRAG      │  │    │    │
│  │  │  └─────────────────┘  └─────────────────┘  │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📦 组件说明

### 核心组件

1. **L1AgentRAG**: 核心代理类，负责协调整个检索流程
2. **QueryComplexityAnalyzer**: 查询复杂度分析器，判断查询的复杂程度
3. **L1AgentRAGTool**: 工具封装类，实现"Agent as Tool"概念

### 检索方法

1. **传统RAG**: 适用于简单的事实查询
   - 基于向量相似度的检索
   - 快速响应
   - 适合直接问答

2. **GraphRAG**: 适用于复杂的关系推理查询
   - 基于知识图谱的检索
   - 支持多跳推理
   - 适合关系分析

## 🚀 快速开始

### 基本使用

```python
import asyncio
from L1_agent_rag import L1AgentRAG

async def main():
    # 创建L1 Agent
    agent = L1AgentRAG()
    
    # 执行查询（自动选择方法）
    result = await agent.query("什么是人工智能？")
    
    print(f"答案: {result.answer}")
    print(f"使用方法: {result.method_used}")
    print(f"复杂度: {result.complexity_analysis.complexity_level}")

asyncio.run(main())
```

### 作为工具使用

```python
import asyncio
import json
from L1_agent_rag import L1AgentRAGTool

async def main():
    # 创建工具
    tool = L1AgentRAGTool()
    
    # 执行查询
    result_json = await tool._arun(
        query="人工智能和机器学习的关系是什么？",
        top_k=5,
        include_metadata=True
    )
    
    # 解析结果
    result = json.loads(result_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))

asyncio.run(main())
```

### 集成到LangGraph Agent

```python
from langchain.tools import BaseTool
from L1_agent_rag import get_l1_agent_rag_tool

# 获取L1 Agent RAG工具
l1_rag_tool = get_l1_agent_rag_tool()

# 添加到LangGraph Agent的工具列表
tools = [
    l1_rag_tool,
    # 其他工具...
]

# 在LangGraph Agent中使用
# agent = LangGraphAgent(tools=tools)
```

## ⚙️ 配置

### 基本配置

```python
from L1_agent_rag import get_config, update_config

# 获取当前配置
config = get_config()

# 更新配置
update_config(
    enable_caching=True,
    cache_ttl=3600
)
```

### 复杂度分析配置

```python
from L1_agent_rag import ComplexityAnalysisConfig

config = ComplexityAnalysisConfig(
    complexity_threshold=0.6,
    rule_based_weight=0.4,
    llm_based_weight=0.6
)
```

### GraphRAG配置

```python
from L1_agent_rag import GraphRAGConfig

config = GraphRAGConfig(
    graph_storage_path="storage/graph_rag",
    max_entities_per_chunk=20,
    max_relationships_per_chunk=15
)
```

## 🔧 高级用法

### 强制使用特定方法

```python
# 强制使用传统RAG
result = await agent.query(
    "什么是Python？", 
    force_method='traditional_rag'
)

# 强制使用GraphRAG
result = await agent.query(
    "Python和Java的关系是什么？", 
    force_method='graph_rag'
)
```

### 构建知识图谱

```python
# 构建知识图谱
success = await agent.build_knowledge_graph()
if success:
    print("知识图谱构建成功")
else:
    print("知识图谱构建失败")
```

### 获取状态信息

```python
# 获取Agent状态
status = agent.get_status()
print(f"GraphRAG可用: {status['graph_rag_available']}")
print(f"总查询数: {status['usage_stats']['total_queries']}")
print(f"传统RAG使用次数: {status['usage_stats']['traditional_rag_used']}")
print(f"GraphRAG使用次数: {status['usage_stats']['graph_rag_used']}")
```

## 📊 性能监控

### 使用统计

```python
# 重置统计
agent.reset_stats()

# 执行一些查询...

# 查看统计
status = agent.get_status()
print(f"使用统计: {status['usage_stats']}")
```

### 执行时间分析

```python
result = await agent.query("查询内容")
print(f"执行时间: {result.execution_time:.2f}秒")
print(f"复杂度分析置信度: {result.complexity_analysis.confidence:.2f}")
```

## 🧪 测试

运行测试脚本：

```bash
python -m L1_agent_rag.test_l1_agent
```

测试包括：
- 复杂度分析测试
- 传统RAG测试
- GraphRAG测试
- 自动选择测试
- 工具接口测试
- 性能测试

## 📝 API参考

### L1AgentRAG

#### 方法

- `query(query: str, top_k: int = 5, force_method: Optional[str] = None) -> L1AgentResult`
  - 执行智能查询
  - 参数：
    - `query`: 用户查询
    - `top_k`: 返回结果数量
    - `force_method`: 强制使用的方法

- `build_knowledge_graph(force_rebuild: bool = False) -> bool`
  - 构建知识图谱
  - 参数：
    - `force_rebuild`: 是否强制重建

- `get_status() -> Dict[str, Any]`
  - 获取Agent状态信息

- `reset_stats() -> None`
  - 重置使用统计

### L1AgentRAGTool

#### 方法

- `_arun(query: str, top_k: int = 5, force_method: Optional[str] = None, include_metadata: bool = True, build_graph_if_missing: bool = False) -> str`
  - 异步执行工具
  - 返回JSON格式的结果

- `get_agent_status() -> Dict[str, Any]`
  - 获取Agent状态

- `build_knowledge_graph(force_rebuild: bool = False) -> bool`
  - 构建知识图谱

### QueryComplexityAnalyzer

#### 方法

- `analyze_complexity(query: str) -> ComplexityAnalysisResult`
  - 分析查询复杂度
  - 返回复杂度分析结果

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可证

MIT License

## 🔗 相关链接

- [GraphRAG论文](https://arxiv.org/abs/2404.16130)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [LangChain文档](https://python.langchain.com/)