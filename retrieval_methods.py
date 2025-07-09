"""检索方法模块

该模块负责各种检索方法的实现，包括：
- 本地RAG检索
- 网络搜索
- MCP服务检索
- 检索结果统一格式化
"""

import os
import json
import hashlib
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# 条件导入faiss模块（解决IDE类型检查问题）
try:
    import faiss
except ImportError:
    faiss = None

from config import config
from database import get_mcp_tools

class BaseRetriever(ABC):
    """检索器基类"""
    
    @abstractmethod
    async def retrieve(self, query: str, **kwargs) -> str:
        """检索方法
        
        参数:
            query: 查询文本
            **kwargs: 其他参数
            
        返回:
            检索结果
        """
        pass

class LocalRetriever(BaseRetriever):
    """本地RAG检索器（简化版本）
    
    注意：此类已被L1 Agent RAG取代，仅保留基本功能以维持兼容性。
    建议使用L1 Agent RAG获得更智能的检索体验。
    """
    
    def __init__(self, document_processor=None, config_instance=None):
        """初始化本地检索器
        
        参数:
            document_processor: 文档处理器实例（可选）
            config_instance: 配置实例，如果为None则使用全局配置
        """
        from config import config as default_config
        
        self.document_processor = document_processor
        self.config = config_instance or default_config
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """本地检索方法
        
        参数:
            query: 查询文本
            **kwargs: 其他参数
            
        返回:
            检索结果
        """
        # 简化的本地检索实现，主要用于兼容性
        try:
            if self.document_processor:
                # 如果有文档处理器，尝试使用它进行检索
                results = await self.document_processor.search_documents(query, top_k=kwargs.get('top_k', 5))
                if results:
                    return "\n\n".join([f"文档: {r.get('title', '未知')}\n内容: {r.get('content', '')}" for r in results])
            
            # 如果没有文档处理器或检索失败，返回提示信息
            return "本地检索器未配置或无可用文档。建议使用L1 Agent RAG进行智能检索。"
            
        except Exception as e:
            return f"本地检索失败: {str(e)}"

class WebSearchRetriever(BaseRetriever):
    """网络搜索检索器"""
    
    def __init__(self):
        """初始化网络搜索检索器"""
        self.bocha_config = config.get_bocha_config()
    
    async def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """执行网络搜索
        
        参数:
            query: 搜索查询
            max_results: 最大结果数
            
        返回:
            搜索结果列表
        """
        import aiohttp
        
        print(f"\n=== WebSearchRetriever.web_search 开始API调用 ===")
        print(f"搜索查询: {query}")
        print(f"最大结果数: {max_results}")
        
        if not self.bocha_config:
            print("错误: Bocha配置未找到")
            return []
        
        if not self.bocha_config.get('api_key'):
            print("错误: BOCHA_API_KEY未配置")
            return []
        
        if not self.bocha_config.get('api_base'):
            print("错误: BOCHA_API_BASE未配置")
            return []
        
        try:
            print(f"API Base URL: {self.bocha_config['api_base']}")
            print(f"API Key存在: {'是' if self.bocha_config.get('api_key') else '否'}")
            print(f"超时设置: {self.bocha_config.get('timeout', 30)}秒")
            
            async with aiohttp.ClientSession() as session:
                # 构建搜索请求 - 注意api_base已经包含了完整路径
                search_url = self.bocha_config['api_base']
                headers = {
                    "Authorization": f"Bearer {self.bocha_config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "query": query,
                    "max_results": max_results,
                    "search_type": "web"
                }
                
                print(f"请求数据: {payload}")
                print("正在发送API请求...")
                
                async with session.post(search_url, json=payload, headers=headers, timeout=self.bocha_config.get('timeout', 30)) as response:
                    print(f"API响应状态码: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"API响应数据类型: {type(data).__name__}")
                        
                        # 提取搜索结果 - Bocha API的响应结构是 data.data.webPages.value
                        if 'data' in data and isinstance(data['data'], dict):
                            web_pages = data['data'].get('webPages', {})
                            if isinstance(web_pages, dict) and 'value' in web_pages:
                                results = web_pages['value']
                                print(f"API返回结果数量: {len(results)}")
                                if results:
                                    print(f"第一个结果预览: {str(results[0])[:200]}...")
                                return results
                        # 兼容其他可能的响应格式
                        results = data.get('results', [])
                        print(f"API返回结果数量: {len(results)}")
                        if results:
                            print(f"第一个结果预览: {str(results[0])[:200]}...")
                        return results
                    else:
                        error_text = await response.text()
                        print(f"网络搜索失败: HTTP {response.status}, 响应: {error_text}")
                        return []
        
        except Exception as e:
            print(f"网络搜索API调用异常: {str(e)}")
            print(f"搜索URL: {search_url if 'search_url' in locals() else '未知'}")
            print(f"API Key: {'已配置' if self.bocha_config.get('api_key') else '未配置'}")
            import traceback
            print(f"异常堆栈: {traceback.format_exc()}")
            return []
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """网络搜索检索
        
        参数:
            query: 查询文本
            **kwargs: 其他参数，包括max_results等
            
        返回:
            搜索结果
        """
        print(f"\n=== WebSearchRetriever 开始网络搜索 ===")
        print(f"查询: {query}")
        
        max_results = kwargs.get('max_results', 5)
        print(f"搜索参数: max_results={max_results}")
        
        try:
            print("正在执行网络搜索...")
            search_results = await self.web_search(query, max_results)
            print(f"搜索完成，找到 {len(search_results)} 个结果")
            
            if not search_results:
                result = "网络搜索未找到相关结果。"
                print(f"网络搜索结果: {result}")
                return result
            
            # 格式化搜索结果
            print("正在格式化搜索结果...")
            result = "网络搜索结果：\n\n"
            for i, item in enumerate(search_results, 1):
                # 兼容不同的字段名（Bocha API使用'name'而不是'title'）
                title = item.get('name', item.get('title', '无标题'))
                url = item.get('url', '')
                snippet = item.get('snippet', '无摘要')
                
                result += f"{i}. **{title}**\n"
                result += f"   链接: {url}\n"
                result += f"   摘要: {snippet}\n\n"
                
                print(f"已处理搜索结果 {i}: {title[:50]}...")
            
            print(f"网络搜索完成，总结果长度: {len(result)}")
            return result
            
        except Exception as e:
            error_msg = f"网络搜索失败: {str(e)}"
            print(f"网络搜索异常: {error_msg}")
            import traceback
            print(f"异常堆栈: {traceback.format_exc()}")
            return error_msg

class MCP_Server(BaseRetriever):
    """MCP服务检索器"""
    
    def __init__(self):
        """初始化MCP服务检索器"""
        self.mcp_tools_cache: Optional[List[Dict]] = None
        self.cache_timestamp = 0
        self.cache_ttl = 300  # 5分钟缓存
    
    async def initialize_mcp_tools(self) -> List[Dict]:
        """初始化MCP工具列表
        
        返回:
            MCP工具列表
        """
        import time
        
        current_time = time.time()
        
        # 检查缓存是否有效
        if (self.mcp_tools_cache is not None and 
            current_time - self.cache_timestamp < self.cache_ttl):
            return self.mcp_tools_cache
        
        try:
            # 从数据库获取MCP工具 - 使用异步方法
            from database import db
            tools_data = await db.get_mcp_tools()
            
            if not tools_data:
                print("数据库中没有找到MCP工具")
                self.mcp_tools_cache = []
                self.cache_timestamp = current_time
                return []
            
            # 组织工具数据
            tools_by_server = {}
            for tool in tools_data:
                server_id = tool['server_id']
                if server_id not in tools_by_server:
                    tools_by_server[server_id] = {
                        'server_id': server_id,
                        'server_name': tool['server_name'],
                        'server_url': tool['server_url'],
                        'tools': []
                    }
                
                tools_by_server[server_id]['tools'].append({
                    'tool_id': tool['tool_id'],
                    'tool_name': tool['tool_name'],
                    'tool_description': tool['tool_description'],
                    'input_schema': tool['input_schema']
                })
            
            self.mcp_tools_cache = list(tools_by_server.values())
            self.cache_timestamp = current_time
            
            print(f"成功加载 {len(self.mcp_tools_cache)} 个MCP服务器的工具")
            return self.mcp_tools_cache
            
        except Exception as e:
            print(f"初始化MCP工具失败: {str(e)}")
            self.mcp_tools_cache = []
            self.cache_timestamp = current_time
            return []
    
    async def call_mcp_service(self, server_url: str, tool_name: str, 
                              arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP服务
        
        参数:
            server_url: 服务器URL
            tool_name: 工具名称
            arguments: 工具参数
            
        返回:
            调用结果
        """
        try:
            # 导入必要的模块
            from fastmcp import Client
            from fastmcp.client.transports import SSETransport
            
            # 使用与test_mcp_service.py相同的方式创建客户端
            async with Client(SSETransport(server_url)) as client:
                # 调用工具
                result = await client.call_tool(tool_name, arguments)
                
                return {
                    'success': True,
                    'result': result,
                    'error': None
                }
            
        except Exception as e:
            error_msg = f"调用MCP服务失败: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'result': None,
                'error': error_msg
            }
    
    async def generate_mcp_answer(self, query: str, conversation_history: List[Dict], 
                                client, model_name: str) -> str:
        """使用MCP服务生成答案
        
        参数:
            query: 用户查询
            conversation_history: 对话历史
            client: OpenAI客户端
            model_name: 模型名称
            
        返回:
            生成的答案
        """
        # 获取MCP工具
        mcp_tools = await self.initialize_mcp_tools()
        
        if not mcp_tools:
            return "当前没有可用的MCP工具。"
        
        # 构建工具描述
        tools_description = "可用的工具：\n"
        for server in mcp_tools:
            tools_description += f"\n服务器: {server['server_name']} ({server['server_url']})\n"
            for tool in server['tools']:
                tools_description += f"  - {tool['tool_name']}: {tool['tool_description']}\n"
        
        # 构建上下文
        context = ""
        if conversation_history:
            context = "对话历史：\n"
            for msg in conversation_history[-3:]:  # 只取最近3轮对话
                role = "用户" if msg['role'] == 'user' else "助手"
                context += f"{role}: {msg['content']}\n"
            context += "\n"
        
        # 构建提示词
        prompt = f"""{context}用户查询: {query}

{tools_description}

请分析用户的查询，判断是否需要使用工具来回答。如果需要使用工具，请按以下格式回复：

USE_TOOL: {{"server_url": "服务器URL", "tool_name": "工具名称", "arguments": {{参数字典}}}}

如果不需要使用工具，请直接回答用户的问题。"""
        
        try:
            # 调用大模型进行工具选择
            model_config = config.get_model_config(model_name)
            if not model_config:
                return "模型配置错误。"
            
            # 构建基础参数
            base_params = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": "你是一个智能助手，能够分析用户查询并选择合适的工具来回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            # 使用统一的参数获取方法
            call_params = config.get_model_call_params(model_name, base_params)
            
            try:
                response = await client.chat.completions.create(**call_params)
            except Exception as e:
                print(f"模型调用失败 - 模型: {model_name}, 错误: {str(e)}")
                print(f"调用参数: {call_params}")
                # 回退到基础参数重试
                basic_params = {k: v for k, v in call_params.items() 
                               if k in ['model', 'messages', 'temperature', 'max_tokens']}
                response = await client.chat.completions.create(**basic_params)
            
            answer = response.choices[0].message.content.strip()
            
            # 检查是否需要使用工具
            if answer.startswith("USE_TOOL:"):
                try:
                    # 解析工具调用信息
                    tool_info_str = answer[9:].strip()
                    tool_info = json.loads(tool_info_str)
                    
                    server_url = tool_info['server_url']
                    tool_name = tool_info['tool_name']
                    arguments = tool_info['arguments']
                    
                    # 调用MCP服务
                    mcp_result = await self.call_mcp_service(server_url, tool_name, arguments)
                    
                    if mcp_result['success']:
                        # 使用工具结果生成最终答案
                        final_prompt = f"""用户查询: {query}

工具调用结果: {mcp_result['result']}

请基于工具调用的结果，为用户提供准确、有用的回答。"""
                        
                        # 构建基础参数
                        final_base_params = {
                            "model": model_config["model"],
                            "messages": [
                                {"role": "system", "content": "你是一个智能助手，能够基于工具调用结果为用户提供准确的回答。"},
                                {"role": "user", "content": final_prompt}
                            ],
                            "temperature": 0.3,
                            "max_tokens": 2000
                        }
                        
                        # 使用统一的参数获取方法
                        final_call_params = config.get_model_call_params(model_name, final_base_params)
                        
                        try:
                            final_response = await client.chat.completions.create(**final_call_params)
                        except Exception as e:
                            print(f"模型调用失败 - 模型: {model_name}, 错误: {str(e)}")
                            print(f"调用参数: {final_call_params}")
                            # 回退到基础参数重试
                            final_basic_params = {k: v for k, v in final_call_params.items() 
                                                  if k in ['model', 'messages', 'temperature', 'max_tokens']}
                            final_response = await client.chat.completions.create(**final_basic_params)
                        
                        return final_response.choices[0].message.content.strip()
                    else:
                        return f"工具调用失败: {mcp_result['error']}"
                        
                except json.JSONDecodeError:
                    return "工具调用格式解析失败。"
                except KeyError as e:
                    return f"工具调用参数缺失: {str(e)}"
            else:
                # 直接返回答案
                return answer
                
        except Exception as e:
            return f"MCP服务调用异常: {str(e)}"
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """MCP服务检索
        
        参数:
            query: 查询文本
            **kwargs: 其他参数，包括conversation_history, client, model_name等
            
        返回:
            检索结果
        """
        conversation_history = kwargs.get('conversation_history', [])
        client = kwargs.get('client')
        model_name = kwargs.get('model_name', config.system_config['default_model'])
        
        if not client:
            return "OpenAI客户端未初始化。"
        
        return await self.generate_mcp_answer(query, conversation_history, client, model_name)

class RetrievalManager:
    """检索管理器"""
    
    def __init__(self, document_processor=None):
        """初始化检索管理器
        
        参数:
            document_processor: 文档处理器实例（可选）
        """
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.document_processor = document_processor
        
        # 注册默认检索器
        self.register_retriever('local', LocalRetriever(document_processor))
        self.register_retriever('web', WebSearchRetriever())
        self.register_retriever('mcp', MCP_Server())
    
    def register_retriever(self, name: str, retriever: BaseRetriever):
        """注册检索器
        
        参数:
            name: 检索器名称
            retriever: 检索器实例
        """
        self.retrievers[name] = retriever
    
    def get_retriever(self, name: str) -> Optional[BaseRetriever]:
        """获取检索器
        
        参数:
            name: 检索器名称
            
        返回:
            检索器实例或None
        """
        return self.retrievers.get(name)
    
    async def retrieve(self, method: str, query: str, **kwargs) -> str:
        """执行检索
        
        参数:
            method: 检索方法名称
            query: 查询文本
            **kwargs: 其他参数
            
        返回:
            检索结果
        """
        print(f"\n=== RetrievalManager 执行检索 ===")
        print(f"检索方法: {method}")
        print(f"查询文本: {query}")
        print(f"其他参数: {kwargs}")
        
        retriever = self.get_retriever(method)
        if not retriever:
            error_msg = f"未找到检索方法: {method}"
            print(f"错误: {error_msg}")
            print(f"可用检索器: {list(self.retrievers.keys())}")
            return error_msg
        
        print(f"找到检索器: {retriever.__class__.__name__}")
        
        try:
            print(f"开始执行 {method} 检索...")
            result = await retriever.retrieve(query, **kwargs)
            print(f"检索完成，结果长度: {len(str(result))}")
            print(f"检索结果预览: {str(result)[:300]}...")
            return result
        except Exception as e:
            error_msg = f"检索失败: {str(e)}"
            print(f"检索异常: {error_msg}")
            import traceback
            print(f"异常堆栈: {traceback.format_exc()}")
            return error_msg
    
    def list_retrievers(self) -> List[str]:
        """列出所有可用的检索器
        
        返回:
            检索器名称列表
        """
        return list(self.retrievers.keys())