"""传统RAG检索实现模块

该模块提供完整的传统RAG检索功能，直接使用document_manager的输出：
- 复用document_manager的向量索引
- 复用document_manager的embedding模型
- 复用document_manager的文档映射关系
- 结果格式化
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 条件导入faiss模块
try:
    import faiss
except ImportError:
    faiss = None

from .config import L1AgentConfig

class TraditionalRAGRetriever:
    """优化后的传统RAG检索器 - 直接使用document_manager的输出"""
    
    def __init__(self, config: L1AgentConfig = None):
        """初始化传统RAG检索器
        
        参数:
            config: L1Agent配置实例
        """
        self.config = config or L1AgentConfig()
        # 直接使用项目的document_processor实例
        from document_manager import document_processor
        self.document_processor = document_processor
    
    def _search_with_existing_index(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """使用document_processor的FAISS索引进行搜索
        
        参数:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        返回:
            距离和索引的元组
        """
        if self.document_processor.index_manager.index is None:
            raise ValueError("document_processor的索引未初始化")
        
        # 确保查询向量是二维的
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 使用document_processor的索引进行搜索
        distances, indices = self.document_processor.index_manager.index.search(query_embedding, top_k)
        return distances, indices
    
    def _get_chunks_from_processor(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """使用document_processor的映射关系获取chunks
        
        参数:
            indices: 索引数组
            
        返回:
            chunks信息列表
        """
        result_chunks = []
        
        for i, idx in enumerate(indices[0]):
            idx = int(idx)
            
            # 检查索引是否有效
            if 0 <= idx < len(self.document_processor.all_chunks):
                chunk_content = self.document_processor.all_chunks[idx]
                
                # 获取文档索引
                doc_idx = self.document_processor.chunks_to_document.get(idx, -1)
                source_file = ""
                if 0 <= doc_idx < len(self.document_processor.doc_sources):
                    source_file = os.path.basename(self.document_processor.doc_sources[doc_idx])
                
                result_chunks.append({
                    "chunk_index": idx,
                    "content": str(chunk_content),
                    "source_file": source_file,
                    "document_index": doc_idx
                })
            else:
                result_chunks.append({
                    "chunk_index": idx,
                    "content": f"未找到chunk {idx}的内容",
                    "source_file": "",
                    "document_index": -1
                })
        
        return result_chunks
    
    def _format_json_output(self, query: str, chunks: List[Dict[str, Any]], 
                           distances: np.ndarray, success: bool = True, 
                           error_message: str = None) -> str:
        """格式化JSON输出
        
        参数:
            query: 查询文本
            chunks: chunks信息列表
            distances: 距离数组
            success: 是否成功
            error_message: 错误信息
            
        返回:
            JSON格式的检索结果
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
    
    def _format_error_output(self, query: str, error_message: str) -> str:
        """格式化错误输出
        
        参数:
            query: 查询文本
            error_message: 错误信息
            
        返回:
            JSON格式的错误结果
        """
        return self._format_json_output(query, [], np.array([[]]), False, error_message)
    
    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> str:
        """使用document_manager的索引进行检索
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数
            
        返回:
            JSON格式的检索结果
        """
        try:
            print(f"\n=== 优化版传统RAG检索开始 ===")
            print(f"查询: {query}")
            print(f"top_k: {top_k}")
            
            # 确保document_processor已初始化
            if not self.document_processor.initialized:
                print("document_processor未初始化，尝试加载索引...")
                success = self.document_processor.load_index_and_mapping()
                if not success:
                    return self._format_error_output(query, "索引未初始化")
                self.document_processor.initialized = True
            
            # 步骤1: 使用document_processor的embedding模型
            print("步骤1: 使用document_processor进行embedding")
            query_embedding = self.document_processor.embedding_model.get_embeddings([query])[0]
            print(f"Embedding维度: {query_embedding.shape}")
            
            # 步骤2: 使用document_processor的FAISS索引搜索
            print(f"步骤2: 使用现有索引搜索相似向量，top_k={top_k}")
            distances, indices = self._search_with_existing_index(query_embedding, top_k)
            print(f"找到 {len(indices[0])} 个相似向量")
            
            # 步骤3: 使用document_processor的映射关系获取chunks
            print("步骤3: 从document_processor获取chunks")
            chunks = self._get_chunks_from_processor(indices)
            print(f"成功获取 {len(chunks)} 个chunks")
            
            # 步骤4: 格式化输出
            print("步骤4: 格式化JSON输出")
            result = self._format_json_output(query, chunks, distances)
            
            print("优化版传统RAG检索完成")
            return result
            
        except Exception as e:
            error_msg = f"传统RAG检索失败: {str(e)}"
            print(f"检索异常: {error_msg}")
            return self._format_error_output(query, error_msg)