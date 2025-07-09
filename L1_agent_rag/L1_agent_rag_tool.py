"""L1 Agent RAG工具模块

该模块将L1AgentRAG封装成一个LangChain工具，可以被LangGraph Agent调用。
实现了"Agent as Tool"的概念。
"""

import json
import asyncio
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
try:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForToolUse,
        CallbackManagerForToolUse,
    )
except ImportError:
    # 兼容不同版本的LangChain
    try:
        from langchain_core.callbacks.manager import (
            AsyncCallbackManagerForToolUse,
            CallbackManagerForToolUse,
        )
    except ImportError:
        # 如果都导入失败，使用基础的CallbackManager
        from langchain.callbacks.manager import (
            AsyncCallbackManager as AsyncCallbackManagerForToolUse,
            CallbackManager as CallbackManagerForToolUse,
        )

from .L1_agent_rag import L1AgentRAG, L1AgentResult
from .L1_agent_rag_query_analyzer import QueryComplexityAnalyzer
from .config import L1AgentConfig

# 获取配置实例
config = L1AgentConfig()
from openai import AsyncOpenAI


class L1AgentRAGInput(BaseModel):
    """L1 Agent RAG工具输入模型"""
    query: str = Field(description="用户查询文本")
    top_k: int = Field(default=5, description="返回结果数量")
    force_method: Optional[str] = Field(
        default=None, 
        description="强制使用的检索方法：'traditional_rag' 或 'graph_rag'，不指定则自动选择"
    )
    include_metadata: bool = Field(default=True, description="是否包含详细的元数据信息")
    build_graph_if_missing: bool = Field(default=False, description="如果知识图谱不存在是否自动构建")


class L1AgentRAGTool(BaseTool):
    """L1 Agent RAG工具
    
    这是一个智能RAG工具，能够：
    1. 自动分析查询复杂度
    2. 选择最适合的检索方法（传统RAG vs GraphRAG）
    3. 执行检索并返回JSON格式的结果
    4. 提供详细的执行信息和元数据
    
    该工具实现了"Agent as Tool"的概念，将复杂的RAG决策逻辑封装在一个工具中。
    """
    
    name: str = "l1_agent_rag"
    description: str = """
智能RAG检索工具。能够根据查询复杂度自动选择最适合的检索方法：
- 对于简单的事实查询，使用传统RAG
- 对于复杂的关系推理查询，使用GraphRAG

输入参数：
- query: 用户查询文本（必需）
- top_k: 返回结果数量（默认5）
- force_method: 强制使用的方法（可选：'traditional_rag' 或 'graph_rag'）
- include_metadata: 是否包含元数据（默认true）
- build_graph_if_missing: 如果知识图谱不存在是否自动构建（默认false）

返回JSON格式的检索结果，包含答案、使用的方法、复杂度分析等信息。
"""
    
    args_schema: Type[BaseModel] = L1AgentRAGInput
    return_direct: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._l1_agent: Optional[L1AgentRAG] = None
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """确保L1 Agent已初始化"""
        if self._initialized and self._l1_agent is not None:
            return
        
        async with self._initialization_lock:
            if self._initialized and self._l1_agent is not None:
                return
            
            try:
                print("初始化L1 Agent RAG...")
                
                # 创建OpenAI客户端
                default_model_config = config.get_model_config(config.default_model)
                client = AsyncOpenAI(
                    api_key=default_model_config["api_key"],
                    base_url=default_model_config["api_base"]
                )
                
                # 初始化组件
                complexity_analyzer = QueryComplexityAnalyzer(client)
                
                # 创建L1 Agent
                self._l1_agent = L1AgentRAG(
                    complexity_analyzer=complexity_analyzer,
                    client=client
                )
                
                self._initialized = True
                print("L1 Agent RAG初始化完成")
                
            except Exception as e:
                print(f"L1 Agent RAG初始化失败: {str(e)}")
                raise
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        force_method: Optional[str] = None,
        include_metadata: bool = True,
        build_graph_if_missing: bool = False,
        run_manager: Optional[CallbackManagerForToolUse] = None,
    ) -> str:
        """同步执行工具（通过异步方法实现）"""
        try:
            # 在新的事件循环中运行异步方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._arun(
                        query=query,
                        top_k=top_k,
                        force_method=force_method,
                        include_metadata=include_metadata,
                        build_graph_if_missing=build_graph_if_missing,
                        run_manager=None
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"L1 Agent RAG工具执行失败: {str(e)}",
                "answer": "抱歉，检索过程中出现错误。",
                "method_used": "error",
                "execution_time": 0.0
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        force_method: Optional[str] = None,
        include_metadata: bool = True,
        build_graph_if_missing: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolUse] = None,
    ) -> str:
        """异步执行工具
        
        参数:
            query: 用户查询
            top_k: 返回结果数量
            force_method: 强制使用的方法
            include_metadata: 是否包含元数据
            build_graph_if_missing: 是否在图谱缺失时自动构建
            run_manager: 回调管理器
            
        返回:
            JSON格式的检索结果
        """
        try:
            # 确保L1 Agent已初始化
            await self._ensure_initialized()
            
            # 如果需要且知识图谱不存在，尝试构建
            if build_graph_if_missing and not self._l1_agent.graph_rag_available:
                print("尝试构建知识图谱...")
                await self._l1_agent.build_knowledge_graph()
            
            # 执行查询
            result: L1AgentResult = await self._l1_agent.query(
                query=query,
                top_k=top_k,
                force_method=force_method
            )
            
            # 构建返回结果
            response_data = {
                "success": result.success,
                "answer": result.answer,
                "method_used": result.method_used,
                "execution_time": result.execution_time
            }
            
            # 添加错误信息（如果有）
            if result.error_message:
                response_data["error"] = result.error_message
            
            # 添加元数据（如果需要）
            if include_metadata:
                response_data["metadata"] = {
                    "complexity_analysis": {
                        "complexity_level": result.complexity_analysis.complexity_level,
                        "confidence": result.complexity_analysis.confidence,
                        "reasoning": result.complexity_analysis.reasoning,
                        "recommended_method": result.complexity_analysis.recommended_method,
                        "features": result.complexity_analysis.features
                    },
                    "agent_status": self._l1_agent.get_status(),
                    "tool_params": {
                        "top_k": top_k,
                        "force_method": force_method,
                        "build_graph_if_missing": build_graph_if_missing
                    }
                }
                
                # 添加其他元数据
                if result.metadata:
                    response_data["metadata"].update(result.metadata)
            
            # 返回JSON格式结果
            return json.dumps(response_data, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"L1 Agent RAG工具执行失败: {str(e)}",
                "answer": "抱歉，检索过程中出现错误。",
                "method_used": "error",
                "execution_time": 0.0
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取L1 Agent状态
        
        返回:
            状态信息字典
        """
        if not self._initialized or self._l1_agent is None:
            return {
                "initialized": False,
                "error": "L1 Agent尚未初始化"
            }
        
        status = self._l1_agent.get_status()
        status["initialized"] = True
        return status
    
    async def build_knowledge_graph(self, force_rebuild: bool = False) -> bool:
        """构建知识图谱
        
        参数:
            force_rebuild: 是否强制重建
            
        返回:
            是否构建成功
        """
        await self._ensure_initialized()
        return await self._l1_agent.build_knowledge_graph(force_rebuild)
    
    def reset_stats(self) -> None:
        """重置使用统计"""
        if self._initialized and self._l1_agent is not None:
            self._l1_agent.reset_stats()
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self._initialized and self._l1_agent is not None:
                await self._l1_agent.cleanup()
                self._l1_agent = None
            self._initialized = False
            print("L1AgentRAGTool资源清理完成")
        except Exception as e:
            print(f"清理L1AgentRAGTool资源时出现异常: {str(e)}")


# 创建工具实例
l1_agent_rag_tool = L1AgentRAGTool()


# 便捷函数
async def create_l1_agent_rag_tool() -> L1AgentRAGTool:
    """创建并初始化L1 Agent RAG工具
    
    返回:
        初始化后的L1AgentRAGTool实例
    """
    tool = L1AgentRAGTool()
    await tool._ensure_initialized()
    return tool


def get_l1_agent_rag_tool() -> L1AgentRAGTool:
    """获取L1 Agent RAG工具实例
    
    返回:
        L1AgentRAGTool实例
    """
    return l1_agent_rag_tool