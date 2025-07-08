"""Agent工具封装模块

该模块将现有的检索方式封装为LangChain标准的Tool，包括：
- 本地文档RAG检索工具
- 联网搜索工具
- MCP服务检索工具
"""

import asyncio
from typing import Dict, Any, Optional, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

import sys
import os
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from retrieval_methods import RetrievalManager
from config import config

# 统一使用绝对导入，避免类型检查问题
from L0_agent_state import ToolResult


class LocalRAGSearchInput(BaseModel):
    """本地RAG检索工具输入"""
    query: str = Field(description="要检索的查询文本")
    top_k: int = Field(default=5, description="返回的文档数量")
    specific_file: Optional[str] = Field(default=None, description="指定文件名")


class InternetSearchInput(BaseModel):
    """联网搜索工具输入"""
    query: str = Field(description="要搜索的查询文本")
    max_results: int = Field(default=5, description="最大结果数量")


class MCPServiceInput(BaseModel):
    """MCP服务工具输入"""
    query: str = Field(description="要查询的文本")
    model_name: str = Field(default="", description="模型名称")


class LocalDocumentRAGTool(BaseTool):
    """本地文档RAG检索工具"""
    name: str = "local_document_rag_search"
    description: str = (
        "用于在企业内部知识库中搜索相关文档和信息。"
        "当问题涉及公司政策、内部流程、技术文档或历史记录时使用。"
        "支持指定特定文件进行检索。"
    )
    args_schema: type = LocalRAGSearchInput
    retrieval_manager: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieval_manager = RetrievalManager()
    
    async def _arun(self, query: str, top_k: int = 5, specific_file: Optional[str] = None) -> str:
        """异步执行本地RAG检索"""
        try:
            result = await self.retrieval_manager.retrieve(
                'local', query,
                top_k=top_k,
                specific_file=specific_file
            )
            return result
        except Exception as e:
            return f"本地检索失败: {str(e)}"
    
    def _run(self, query: str, top_k: int = 5, specific_file: Optional[str] = None) -> str:
        """同步执行本地RAG检索"""
        try:
            # 创建新的事件循环来运行异步方法
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.retrieval_manager.retrieve(
                        'local', query,
                        top_k=top_k,
                        specific_file=specific_file
                    )
                )
                return future.result()
        except Exception as e:
            return f"本地检索失败: {str(e)}"


class InternetSearchTool(BaseTool):
    """联网搜索工具"""
    name: str = "internet_search"
    description: str = (
        "用于在公共互联网上搜索最新的信息、新闻、技术趋势或任何企业内部知识库未覆盖的通用知识。"
        "当问题涉及实时性、外部事件或通用概念时使用。"
    )
    args_schema: type = InternetSearchInput
    retrieval_manager: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieval_manager = RetrievalManager()
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        """异步执行联网搜索"""
        try:
            result = await self.retrieval_manager.retrieve(
                'web', query,
                max_results=max_results
            )
            return result
        except Exception as e:
            return f"联网搜索失败: {str(e)}"
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """同步执行联网搜索"""
        try:
            # 创建新的事件循环来运行异步方法
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.retrieval_manager.retrieve(
                        'web', query,
                        max_results=max_results
                    )
                )
                return future.result()
        except Exception as e:
            return f"联网搜索失败: {str(e)}"


class MCPServiceTool(BaseTool):
    """MCP服务检索工具"""
    name: str = "mcp_service_lookup"
    description: str = (
        "用于从MCP (Model Context Protocol) 服务中查询数据。"
        "用于高精度Excel文档检索，可以精确查询Excel表格中的数据内容。"
        "当需要从Excel文档中获取特定数据、进行数据分析或查询表格信息时使用。"
    )
    args_schema: type = MCPServiceInput
    retrieval_manager: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieval_manager = RetrievalManager()
    
    async def _arun(self, query: str, conversation_history: List[Dict] = None, 
                   model_name: str = None) -> str:
        """异步执行MCP服务检索"""
        client = None
        try:
            if conversation_history is None:
                conversation_history = []
            if model_name is None:
                model_name = config.system_config['default_model']
            
            # 创建OpenAI客户端
            from openai import AsyncOpenAI
            model_config = config.get_model_config(model_name)
            client = AsyncOpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['api_base']
            )
            
            result = await self.retrieval_manager.retrieve(
                'mcp', query,
                conversation_history=conversation_history,
                client=client,
                model_name=model_name
            )
            return result
        except Exception as e:
            return f"MCP服务检索失败: {str(e)}"
        finally:
            # 确保客户端正确关闭
            if client is not None:
                try:
                    # AsyncOpenAI使用close()方法而不是aclose()
                    await client.close()
                except Exception as close_error:
                    print(f"关闭AsyncOpenAI客户端时出错: {close_error}")
    
    def _run(self, query: str, conversation_history: List[Dict] = None, 
            model_name: str = None) -> str:
        """同步执行MCP服务检索"""
        try:
            if conversation_history is None:
                conversation_history = []
            if model_name is None:
                model_name = config.system_config['default_model']
            
            # 创建新的事件循环来运行异步方法
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._arun(query, conversation_history, model_name)
                )
                return future.result()
        except Exception as e:
            return f"MCP服务检索失败: {str(e)}"


class AgentToolManager:
    """Agent工具管理器"""
    
    def __init__(self):
        """初始化工具管理器"""
        self.retrieval_manager = RetrievalManager()
        self.tools = self._create_tools()
    
    def _create_tools(self) -> Dict[str, BaseTool]:
        """创建所有工具
        
        返回:
            工具字典
        """
        return {
            "local_document_rag_search": LocalDocumentRAGTool(),
            "internet_search": InternetSearchTool(),
            "mcp_service_lookup": MCPServiceTool()
        }
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取指定工具
        
        参数:
            tool_name: 工具名称
            
        返回:
            工具实例或None
        """
        return self.tools.get(tool_name)
    
    def get_available_tools(self, used_tools=None) -> Dict[str, BaseTool]:
        """获取可用工具（排除已使用的）
        
        参数:
            used_tools: 已使用的工具集合或列表，可选
            
        返回:
            可用工具字典
        """
        if used_tools is None:
            # 如果没有传递used_tools，返回所有工具的描述信息
            return {
                name: {'description': tool.description} 
                for name, tool in self.tools.items()
            }
        
        # 兼容list和set类型的used_tools
        used_tools_set = set(used_tools) if isinstance(used_tools, list) else used_tools
        return {
            name: tool for name, tool in self.tools.items() 
            if name not in used_tools_set
        }
    
    def get_tool_descriptions(self, used_tools: set) -> str:
        """获取可用工具的描述
        
        参数:
            used_tools: 已使用的工具集合
            
        返回:
            工具描述字符串
        """
        available_tools = self.get_available_tools(used_tools)
        
        if not available_tools:
            return "没有可用的工具。"
        
        descriptions = []
        for name, tool in available_tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """同步执行指定工具
        
        参数:
            tool_name: 工具名称
            params: 工具参数字典
            
        返回:
            工具执行结果
        """
        print(f"\n=== 工具管理器执行工具 ===")
        print(f"工具名称: {tool_name}")
        print(f"工具参数: {params}")
        
        tool = self.get_tool(tool_name)
        if not tool:
            error_msg = f"工具 {tool_name} 不存在"
            print(f"错误: {error_msg}")
            return ToolResult(
                success=False,
                content="",
                error=error_msg,
                tool_name=tool_name
            )
        
        print(f"找到工具: {tool.__class__.__name__}")
        
        try:
            # 使用同步方法执行工具
            print(f"开始执行工具 {tool_name}...")
            result = tool._run(**params)
            print(f"工具执行完成，结果长度: {len(str(result))}")
            print(f"工具执行结果预览: {str(result)[:200]}...")
            
            return ToolResult(
                success=True,
                content=result,
                tool_name=tool_name
            )
        except Exception as e:
            error_msg = str(e)
            print(f"工具执行异常: {error_msg}")
            return ToolResult(
                success=False,
                content="",
                error=error_msg,
                tool_name=tool_name
            )
    
    async def execute_tool_async(self, tool_name: str, **kwargs) -> ToolResult:
        """异步执行指定工具
        
        参数:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        返回:
            工具执行结果
        """
        print(f"\n=== 工具管理器异步执行工具 ===")
        print(f"工具名称: {tool_name}")
        print(f"工具参数: {kwargs}")
        
        tool = self.get_tool(tool_name)
        if not tool:
            error_msg = f"工具 {tool_name} 不存在"
            print(f"错误: {error_msg}")
            return ToolResult(
                success=False,
                content="",
                error=error_msg,
                tool_name=tool_name
            )
        
        print(f"找到工具: {tool.__class__.__name__}")
        
        try:
            # 使用异步方法执行工具
            print(f"开始执行工具 {tool_name}...")
            result = await tool._arun(**kwargs)
            print(f"工具执行完成，结果长度: {len(str(result))}")
            print(f"工具执行结果预览: {str(result)[:200]}...")
            
            return ToolResult(
                success=True,
                content=result,
                tool_name=tool_name
            )
        except Exception as e:
            error_msg = str(e)
            print(f"工具执行异常: {error_msg}")
            return ToolResult(
                success=False,
                content="",
                error=error_msg,
                tool_name=tool_name
            )