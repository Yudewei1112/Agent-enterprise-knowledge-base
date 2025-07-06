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
    """本地RAG检索器"""
    
    # 类级别的模型缓存
    _cached_model = None
    _cached_model_path = None
    
    @classmethod
    def clear_model_cache(cls):
        """清除类级别的模型缓存
        
        在需要重新加载模型时调用此方法
        """
        cls._cached_model = None
        cls._cached_model_path = None
        print("已清除embedding模型缓存")
    
    def __init__(self, document_processor=None, config_instance=None):
        """初始化本地检索器
        
        参数:
            document_processor: 文档处理器实例（可选，主要用于获取embedding模型）
            config_instance: 配置实例，如果为None则使用全局配置
        """
        from config import config as default_config
        
        self.document_processor = document_processor
        self.config = config_instance or default_config
        
        # 从配置获取并规范化路径
        self._load_paths_from_config()
        
        # 缓存配置参数
        self._cache_config_params()
        
        # 验证配置
        self._validate_config()
        
        # 初始化状态变量
        self.faiss_index = None
        self.chunks_mapping = None
        self.model = None
    
    def _load_paths_from_config(self) -> None:
        """从配置加载并规范化路径"""
        # 验证system_config是否可用
        if not hasattr(self.config, 'system_config') or self.config.system_config is None:
            raise ValueError("配置实例缺少有效的system_config属性")
        
        # 获取基础路径配置
        self.faiss_index_path = self.config.system_config.get('index_file', 'storage/Faiss/faiss_index.faiss')
        self.chunks_mapping_path = self.config.system_config.get('chunks_mapping_file', 'storage/Faiss/chunks_mapping.npy')
        self.chunks_dir = self.config.system_config.get('chunks_dir', 'chunks')
        
        # 规范化路径（转换为绝对路径）
        self.faiss_index_path = os.path.abspath(self.faiss_index_path)
        self.chunks_mapping_path = os.path.abspath(self.chunks_mapping_path)
        self.chunks_dir = os.path.abspath(self.chunks_dir)
    
    def _cache_config_params(self) -> None:
        """缓存常用配置参数"""
        # 验证system_config是否可用
        if not hasattr(self.config, 'system_config') or self.config.system_config is None:
            raise ValueError("配置实例缺少有效的system_config属性")
        
        self.local_model_path = self.config.system_config.get('local_model_path', 'local_m3e_model')
        self.max_chunk_chars = self.config.system_config.get('max_chunk_chars', 500)
        self.chunk_overlap = self.config.system_config.get('chunk_overlap', 100)
        self.embedding_cache_size = self.config.system_config.get('embedding_cache_size', 1000)
        
        # 缓存目录配置
        self.cache_dir = self.config.system_config.get('cache_dir', 'cache')
        self.upload_dir = self.config.system_config.get('upload_dir', 'uploads')
    
    def _validate_config(self) -> None:
        """验证配置参数
        
        抛出:
            ValueError: 配置参数无效时
        """
        if not self.config:
            raise ValueError("配置实例不能为空")
        
        if not hasattr(self.config, 'system_config'):
            raise ValueError("配置实例缺少system_config属性")
        
        # 验证路径配置
        if not self.faiss_index_path:
            raise ValueError("Faiss索引文件路径不能为空")
        
        if not self.chunks_mapping_path:
            raise ValueError("Chunks映射文件路径不能为空")
        
        # 验证模型配置
        if not hasattr(self.config, 'system_config') or self.config.system_config is None:
            raise ValueError("配置实例缺少有效的system_config属性")
        
        if not hasattr(self.config.system_config, 'get'):
            raise ValueError("system_config必须是字典类型")
        
        # 验证本地模型路径配置
        local_model_path = self.config.system_config.get('local_model_path')
        if local_model_path and not isinstance(local_model_path, str):
            raise ValueError("local_model_path必须是字符串类型")
        
        # 验证chunks目录配置
        chunks_dir = self.config.system_config.get('chunks_dir', 'chunks')
        if not isinstance(chunks_dir, str):
            raise ValueError("chunks_dir必须是字符串类型")
        
        # 验证存储目录配置
        storage_dir = os.path.dirname(self.faiss_index_path)
        if storage_dir and not os.path.exists(storage_dir):
            try:
                os.makedirs(storage_dir, exist_ok=True)
            except Exception as e:
                raise ValueError(f"无法创建存储目录 {storage_dir}: {str(e)}")
    
    def reload_config(self, new_config=None) -> None:
        """重新加载配置
        
        参数:
            new_config: 新的配置实例，如果为None则重新加载当前配置
        """
        if new_config:
            self.config = new_config
        
        # 重新加载路径和参数
        self._load_paths_from_config()
        self._cache_config_params()
        
        # 重新验证配置
        self._validate_config()
        
        # 清除缓存的索引和映射，强制重新加载
        self.faiss_index = None
        self.chunks_mapping = None
        
        print(f"配置已重新加载: {self.faiss_index_path}")
    
    def get_config_summary(self) -> dict:
        """获取当前配置摘要
        
        返回:
            配置摘要字典
        """
        return {
            'faiss_index_path': self.faiss_index_path,
            'chunks_mapping_path': self.chunks_mapping_path,
            'chunks_dir': self.chunks_dir,
            'local_model_path': self.local_model_path,
            'max_chunk_chars': self.max_chunk_chars,
            'chunk_overlap': self.chunk_overlap,
            'cache_dir': self.cache_dir,
            'upload_dir': self.upload_dir
        }
    
    def _check_required_files(self) -> bool:
        """检查必需的文件是否存在
        
        返回:
            bool: 文件是否都存在
        """
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"Faiss索引文件不存在: {self.faiss_index_path}")
        
        if not os.path.exists(self.chunks_mapping_path):
            raise FileNotFoundError(f"Chunks映射文件不存在: {self.chunks_mapping_path}")
        
        return True
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """步骤1: 对用户问题进行embedding
        
        参数:
            query: 用户查询文本
            
        返回:
            np.ndarray: 查询的embedding向量
        """
        if self.document_processor is None or self.document_processor.model is None:
            # 如果没有document_processor，使用类级别的模型缓存
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 检查类级别缓存是否可用且路径匹配
            if (LocalRetriever._cached_model is None or 
                LocalRetriever._cached_model_path != self.local_model_path):
                print("加载embedding模型...")
                local_model_path = self.local_model_path
                if os.path.exists(local_model_path):
                    LocalRetriever._cached_model = SentenceTransformer(local_model_path)
                    print(f"已加载本地模型: {local_model_path}")
                else:
                    LocalRetriever._cached_model = SentenceTransformer('moka-ai/m3e-base')
                    print("已加载在线模型: moka-ai/m3e-base")
                
                # 更新缓存路径
                LocalRetriever._cached_model_path = local_model_path
                
                # 尝试移动到GPU
                if torch.cuda.is_available():
                    print("CUDA可用，尝试将模型移动到GPU...")
                    try:
                        LocalRetriever._cached_model = LocalRetriever._cached_model.to('cuda')
                        print(f"模型已成功移动到GPU: {LocalRetriever._cached_model.device}")
                    except Exception as e:
                        print(f"模型移动到GPU失败，使用CPU。错误: {e}")
                else:
                    print("CUDA不可用，模型将使用CPU进行embedding")
                print(f"当前模型设备: {LocalRetriever._cached_model.device}")
            else:
                print("使用缓存的embedding模型")
            
            # 执行embedding
            import time
            start_time = time.time()
            embedding = LocalRetriever._cached_model.encode(query, convert_to_tensor=False)
            encode_time = time.time() - start_time
            print(f"Embedding完成，耗时: {encode_time:.4f}秒")
            
            return embedding
        else:
            return self.document_processor.get_query_embedding_cached(query)
    
    def _search_similar_vectors(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """步骤2: 与Faiss向量数据库进行相似度计算（兼容模式：使用已优化的索引）
        
        参数:
            query_embedding: 查询的embedding向量
            top_k: 返回的相似向量数量
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (distances, indices)
        """
        import faiss
        
        # 优先使用document_processor中已经优化的索引（可能在GPU上）
        if (self.document_processor is not None and 
            hasattr(self.document_processor, 'index') and 
            self.document_processor.index is not None):
            index = self.document_processor.index
            print("使用已优化的FAISS索引进行检索")
        else:
            # 如果没有可用的优化索引，则加载并尝试优化
            index = self._load_and_optimize_index()
        
        # 确保查询向量是二维的
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 执行相似度搜索
        distances, indices = index.search(query_embedding, top_k)
        
        return distances, indices
    
    def _load_and_optimize_index(self) -> faiss.Index:
        """加载并优化FAISS索引（兼容模式：优先使用GPU，失败时回退到CPU）
        
        返回:
            优化后的FAISS索引
        """
        if faiss is None:
            raise ImportError("faiss模块未安装，请安装faiss-cpu或faiss-gpu")
        
        # 加载CPU索引
        cpu_index = faiss.read_index(self.faiss_index_path)
        
        # 尝试转移到GPU
        gpu_index = self._try_move_to_gpu(cpu_index)
        if gpu_index is not None:
            print("检索时FAISS索引已成功转移到GPU")
            return gpu_index
        else:
            print("检索时FAISS索引使用CPU模式")
            return cpu_index
    
    def _try_move_to_gpu(self, index: faiss.Index) -> Optional[faiss.Index]:
        """尝试将FAISS索引转移到GPU（兼容模式）
        
        参数:
            index: CPU上的FAISS索引
            
        返回:
            GPU索引（成功时）或None（失败时）
        """
        try:
            import torch
            
            # 检查faiss模块是否可用
            if faiss is None:
                return None
            
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                return None
            
            # 检查是否有faiss-gpu支持
            if not hasattr(faiss, 'StandardGpuResources'):
                return None
            
            # 尝试创建GPU资源
            gpu_res = faiss.StandardGpuResources()
            
            # 尝试转移到GPU
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            
            # 验证GPU索引是否工作正常
            if gpu_index.ntotal != index.ntotal:
                return None
                
            return gpu_index
            
        except Exception:
            return None
    
    def _get_chunks_from_indices(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """步骤3: 根据chunks_mapping.npy获取对应的chunks
        
        参数:
            indices: Faiss搜索返回的索引数组
            
        返回:
            List[Dict]: 包含chunk信息的列表
        """
        # 加载chunks映射数据
        mapping_data = np.load(self.chunks_mapping_path, allow_pickle=True).item()
        
        chunks_to_document = mapping_data.get('chunks_to_document', {})
        doc_sources = mapping_data.get('doc_sources', [])
        
        print(f"调试信息: 文档源列表有{len(doc_sources)}个文档")
        print(f"调试信息: chunks_to_document映射有{len(chunks_to_document)}个条目")
        
        # 构建全局chunk索引到具体文件和chunk的映射
        global_chunk_index = 0
        chunk_index_to_info = {}  # {global_index: {'file_path': str, 'local_index': int, 'doc_index': int}}
        
        for doc_idx, source in enumerate(doc_sources):
            filename = os.path.basename(source)
            # 处理特殊字符，生成对应的JSON文件名（与document_manager.py中的逻辑保持一致）
            import re
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
            json_filename = f"{safe_filename}.json"
            chunk_file_path = os.path.join("chunks", json_filename)
            
            if os.path.exists(chunk_file_path):
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunks = chunk_data.get('chunks', [])
                    
                    # 为每个chunk建立映射
                    for local_idx, chunk in enumerate(chunks):
                        chunk_index_to_info[global_chunk_index] = {
                            'file_path': chunk_file_path,
                            'local_index': local_idx,
                            'doc_index': doc_idx,
                            'source_file': filename,
                            'chunk_content': chunk
                        }
                        global_chunk_index += 1
        
        print(f"调试信息: 构建了{len(chunk_index_to_info)}个全局chunk索引映射")
        
        # 获取对应的chunks
        result_chunks = []
        for i, idx in enumerate(indices[0]):
            idx = int(idx)
            print(f"调试信息: 处理chunk索引 {idx}")
            
            if idx in chunk_index_to_info:
                chunk_info = chunk_index_to_info[idx]
                chunk_content = chunk_info['chunk_content']
                
                # 确保content是字符串
                if isinstance(chunk_content, dict):
                    if 'content' in chunk_content:
                        chunk_text = chunk_content['content']
                    else:
                        chunk_text = str(chunk_content)
                else:
                    chunk_text = str(chunk_content)
                
                result_chunks.append({
                    "chunk_index": idx,
                    "content": chunk_text,
                    "source_file": chunk_info['source_file'],
                    "document_index": chunk_info['doc_index']
                })
                
                print(f"调试信息: chunk {idx} -> 文档{chunk_info['doc_index']} ({chunk_info['source_file']})")
            else:
                print(f"警告: chunk索引 {idx} 未找到对应的映射")
                # 使用原有的映射方式作为备用
                doc_idx = chunks_to_document.get(idx, -1)
                source_file = ""
                if 0 <= doc_idx < len(doc_sources):
                    source_file = os.path.basename(doc_sources[doc_idx])
                
                result_chunks.append({
                    "chunk_index": idx,
                    "content": f"未找到chunk {idx}的内容",
                    "source_file": source_file,
                    "document_index": doc_idx
                })
        
        return result_chunks
    
    def _format_json_output(self, query: str, chunks: List[Dict[str, Any]], 
                           distances: np.ndarray, success: bool = True, 
                           error_message: str = None) -> str:
        """步骤4: 严格以JSON格式输出结果
        
        参数:
            query: 原始查询
            chunks: 检索到的chunks列表
            distances: 相似度距离
            success: 是否成功
            error_message: 错误信息（如果有）
            
        返回:
            str: JSON格式的检索结果
        """
        if not success:
            result = {
                "success": False,
                "error": error_message,
                "query": query,
                "results": []
            }
        else:
            results = []
            for i, chunk in enumerate(chunks):
                # 计算相似度分数（距离越小，相似度越高）
                similarity_score = 1.0 / (1.0 + float(distances[0][i])) if i < len(distances[0]) else 0.0
                
                results.append({
                    "rank": i + 1,
                    "chunk_index": chunk["chunk_index"],
                    "similarity_score": round(similarity_score, 4),
                    "distance": float(distances[0][i]) if i < len(distances[0]) else 0.0,
                    "content": chunk["content"],
                    "source_file": chunk["source_file"],
                    "document_index": chunk["document_index"]
                })
            
            result = {
                "success": True,
                "query": query,
                "total_results": len(results),
                "results": results
            }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """本地检索主方法
        
        按照4个步骤执行检索：
        1. 对用户问题进行embedding
        2. 与Faiss向量数据库进行相似度计算
        3. 取排名前top_k = 3，并根据chunks_mapping.npy获取对应的chunks
        4. 严格以JSON格式输出
        
        参数:
            query: 查询文本
            **kwargs: 其他参数，包括top_k等
            
        返回:
            str: JSON格式的检索结果
        """
        top_k = kwargs.get('top_k', 3)  # 默认取前3个结果
        
        try:
            # 检查必需文件
            self._check_required_files()
            
            # 步骤1: 对用户问题进行embedding
            print(f"步骤1: 对查询进行embedding - {query}")
            query_embedding = self._get_query_embedding(query)
            print(f"Embedding维度: {query_embedding.shape}")
            
            # 步骤2: 与Faiss向量数据库进行相似度计算
            print(f"步骤2: 在Faiss索引中搜索相似向量，top_k={top_k}")
            distances, indices = self._search_similar_vectors(query_embedding, top_k)
            print(f"找到 {len(indices[0])} 个相似向量")
            
            # 步骤3: 根据chunks_mapping.npy获取对应的chunks
            print(f"步骤3: 获取对应的chunks")
            chunks = self._get_chunks_from_indices(indices)
            print(f"成功获取 {len(chunks)} 个chunks")
            
            # 步骤4: 严格以JSON格式输出
            print(f"步骤4: 格式化JSON输出")
            result = self._format_json_output(query, chunks, distances)
            
            return result
            
        except Exception as e:
            error_msg = f"本地检索失败: {str(e)}"
            print(f"检索异常: {error_msg}")
            
            # 即使出错也返回JSON格式
            return self._format_json_output(query, [], np.array([[]]), False, error_msg)

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