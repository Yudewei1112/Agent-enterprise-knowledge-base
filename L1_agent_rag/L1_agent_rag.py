"""L1 Agent RAG核心模块

L1智能体-RAG，能够根据查询复杂度自动选择最适合的检索方法：
- 传统RAG：适用于简单的事实查询
- GraphRAG：适用于复杂的关系推理查询
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import AsyncOpenAI

# 导入现有模块
from .config import L1AgentConfig

# 获取配置实例
config = L1AgentConfig()
# 导入本模块组件
from .L1_agent_rag_query_analyzer import QueryComplexityAnalyzer, ComplexityAnalysisResult
from .graphrag import GraphRAGRetriever, MultiGraphRAGManager
from .traditional_rag import TraditionalRAGRetriever


@dataclass
class L1AgentResult:
    """L1 Agent执行结果"""
    success: bool
    answer: str
    method_used: str  # 'traditional_rag' 或 'graph_rag'
    complexity_analysis: ComplexityAnalysisResult
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class L1AgentRAG:
    """L1 智能RAG代理
    
    这个代理能够：
    1. 自动分析查询复杂度
    2. 选择最适合的检索方法（传统RAG vs GraphRAG）
    3. 执行检索并返回结果
    4. 提供详细的执行信息和元数据
    """
    
    def __init__(self, 
                 complexity_analyzer: QueryComplexityAnalyzer = None,
                 client: AsyncOpenAI = None):
        """初始化L1 Agent RAG
        
        参数:
            complexity_analyzer: 复杂度分析器
            client: OpenAI客户端
        """
        self.client = client or self._create_client()
        
        # 初始化组件
        self.complexity_analyzer = complexity_analyzer or QueryComplexityAnalyzer(self.client)
        
        # 初始化检索器
        self.traditional_rag = TraditionalRAGRetriever(config)
        self.graph_rag = None
        self.graph_rag_available = False
        
        # 初始化多图谱GraphRAG
        self._initialize_multi_graph_rag()
        
        # 统计信息
        self.usage_stats = {
            'total_queries': 0,
            'traditional_rag_used': 0,
            'graph_rag_used': 0,
            'fallback_count': 0
        }
    
    def _create_client(self) -> AsyncOpenAI:
        """创建OpenAI客户端"""
        # 获取配置实例
        from .config import get_config
        config_instance = get_config()
        default_model_config = config_instance.get_model_config(config_instance.default_model)
        return AsyncOpenAI(
            api_key=default_model_config["api_key"],
            base_url=default_model_config["api_base"]
        )
    
    def _initialize_multi_graph_rag(self) -> None:
        """初始化多图谱GraphRAG组件"""
        try:
            self.multi_graph_manager = MultiGraphRAGManager()
            
            # 检查是否存在已构建的知识图谱
            if self.multi_graph_manager.graph_builders:
                self.graph_rag_available = True
                print(f"多图谱GraphRAG加载成功，共{len(self.multi_graph_manager.graph_builders)}个图谱")
            else:
                self.graph_rag_available = False
                print("多图谱GraphRAG未找到，需要构建")
                
        except Exception as e:
            print(f"多图谱GraphRAG初始化失败: {str(e)}")
            self.graph_rag_available = False
    
    async def build_knowledge_graph(self, force_rebuild: bool = False) -> bool:
        """构建多图谱知识图谱"""
        try:
            print("开始多图谱知识图谱构建流程...")
            
            # 检查多图谱管理器是否已初始化
            if not hasattr(self, 'multi_graph_manager') or self.multi_graph_manager is None:
                print("多图谱管理器未初始化，尝试重新初始化...")
                self._initialize_multi_graph_rag()
                if not hasattr(self, 'multi_graph_manager') or self.multi_graph_manager is None:
                    print("多图谱管理器初始化失败")
                    return False
            
            # 使用多图谱管理器构建图谱
            success = await self.multi_graph_manager.build_graphs_from_chunks(force_rebuild)
            
            if success:
                self.graph_rag_available = True
                print(f"多图谱知识图谱构建成功，共构建{len(self.multi_graph_manager.graph_builders)}个图谱")
                return True
            else:
                print("多图谱知识图谱构建失败")
                return False
            
        except Exception as e:
            print(f"多图谱知识图谱构建失败: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    async def _execute_graph_rag(self, query: str, top_k: int, **kwargs) -> str:
        """执行多图谱GraphRAG检索"""
        if not self.graph_rag_available or not self.multi_graph_manager:
            return "多图谱GraphRAG功能不可用，请先构建知识图谱。"
        
        try:
            print("执行多图谱GraphRAG检索...")
            # 使用多图谱管理器进行查询
            result = await self.multi_graph_manager.query(
                query=query,
                max_graphs=3,  # 最多查询3个图谱
                top_k_per_graph=top_k
            )
            return result
            
        except Exception as e:
            error_msg = f"多图谱GraphRAG检索失败: {str(e)}"
            print(error_msg)
            return error_msg
    
    async def _execute_traditional_rag(self, query: str, top_k: int, **kwargs) -> str:
        """执行传统RAG检索（完整实现）
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数
            
        返回:
            检索结果
        """
        try:
            print("执行传统RAG检索...")
            
            # 调用传统RAG检索器
            rag_result = await self.traditional_rag.retrieve(query, top_k, **kwargs)
            
            # 解析JSON结果
            import json
            result_data = json.loads(rag_result)
            
            if result_data.get('success', False):
                # 基于检索结果生成答案
                answer = await self._generate_answer_from_traditional_rag(query, result_data)
                return answer
            else:
                error_msg = result_data.get('error', '未知错误')
                return f"传统RAG检索失败: {error_msg}"
                
        except Exception as e:
            error_msg = f"传统RAG执行异常: {str(e)}"
            print(error_msg)
            return error_msg
    
    async def _generate_answer_from_traditional_rag(self, query: str, rag_result: Dict) -> str:
        """基于传统RAG结果生成答案（完整实现）
        
        参数:
            query: 原始查询
            rag_result: RAG检索结果
            
        返回:
            生成的答案
        """
        try:
            results = rag_result.get('results', [])
            
            if not results:
                return "抱歉，没有找到相关信息来回答您的问题。"
            
            # 构建上下文
            context_parts = []
            for i, result in enumerate(results[:3]):  # 只使用前3个最相关的结果
                content = result.get('content', '')
                source = result.get('source_file', '未知来源')
                similarity = result.get('similarity_score', 0.0)
                
                context_parts.append(f"[来源{i+1}: {source}, 相似度: {similarity:.3f}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # 构建提示词
            prompt = f"""基于以下检索到的相关信息，请回答用户的问题。

用户问题：{query}

相关信息：
{context}

请根据上述信息提供准确、有用的回答。如果信息不足以回答问题，请说明。"""
            
            # 调用LLM生成答案
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的AI助手，能够基于提供的信息准确回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 添加来源信息
            sources = list(set([result.get('source_file', '未知来源') for result in results[:3]]))
            source_info = f"\n\n参考来源：{', '.join(sources)}"
            
            return answer + source_info
            
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return f"基于检索结果生成答案时出现错误: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """获取L1 Agent状态信息
        
        返回:
            状态信息字典
        """
        return {
            'graph_rag_available': self.graph_rag_available,
            'traditional_rag_available': self.traditional_rag is not None,
            'usage_stats': self.usage_stats.copy(),
            'components': {
                'traditional_rag': self.traditional_rag is not None,
                'graph_rag': self.graph_rag is not None,
                'complexity_analyzer': self.complexity_analyzer is not None
            }
        }
    
    def reset_stats(self) -> None:
        """重置使用统计"""
        self.usage_stats = {
            'total_queries': 0,
            'traditional_rag_used': 0,
            'graph_rag_used': 0,
            'fallback_count': 0
        }
        print("使用统计已重置")
    
    async def cleanup(self) -> None:
        """清理资源，确保所有异步任务都被正确处理"""
        try:
            print("正在清理L1AgentRAG资源...")
            
            # 清理OpenAI客户端
            if hasattr(self.client, 'close'):
                await self.client.close()
            
            # 清理多图谱GraphRAG相关资源
            if hasattr(self, 'multi_graph_manager') and self.multi_graph_manager:
                for builder in self.multi_graph_manager.graph_builders.values():
                    if hasattr(builder, 'extractor') and builder.extractor:
                        if hasattr(builder.extractor.client, 'close'):
                            await builder.extractor.client.close()
            
            # 清理复杂度分析器
            if hasattr(self.complexity_analyzer, 'client') and hasattr(self.complexity_analyzer.client, 'close'):
                await self.complexity_analyzer.client.close()
            
            print("L1AgentRAG资源清理完成")
            
        except Exception as e:
            print(f"清理L1AgentRAG资源时出现异常: {str(e)}")
    
    async def query(self, 
                   query: str, 
                   top_k: int = 5,
                   force_method: Optional[str] = None,
                   **kwargs) -> L1AgentResult:
        """执行智能查询
        
        参数:
            query: 用户查询
            top_k: 返回结果数量
            force_method: 强制使用的方法 ('traditional_rag' 或 'graph_rag')
            **kwargs: 其他参数
            
        返回:
            L1Agent执行结果
        """
        import time
        start_time = time.time()
        
        self.usage_stats['total_queries'] += 1
        
        try:
            # 1. 分析查询复杂度
            print(f"\n=== L1 Agent RAG 查询处理 ===")
            print(f"查询: {query}")
            
            if force_method:
                print(f"强制使用方法: {force_method}")
                complexity_analysis = ComplexityAnalysisResult(
                    complexity_level='complex' if force_method == 'graph_rag' else 'simple',
                    confidence=1.0,
                    reasoning=f"用户强制指定使用{force_method}",
                    recommended_method=force_method,
                    features={}
                )
            else:
                print("分析查询复杂度...")
                complexity_analysis = await self.complexity_analyzer.analyze_complexity(query)
                print(f"复杂度分析结果: {self.complexity_analyzer.get_analysis_summary(complexity_analysis)}")
            
            # 2. 选择检索方法
            selected_method = complexity_analysis.recommended_method
            
            # 如果推荐GraphRAG但不可用，回退到传统RAG
            if selected_method == 'graph_rag' and not self.graph_rag_available:
                print("GraphRAG不可用，回退到传统RAG")
                selected_method = 'traditional_rag'
                self.usage_stats['fallback_count'] += 1
            
            print(f"选择的检索方法: {selected_method}")
            
            # 3. 执行检索
            if selected_method == 'graph_rag':
                answer = await self._execute_graph_rag(query, top_k, **kwargs)
                self.usage_stats['graph_rag_used'] += 1
            else:
                answer = await self._execute_traditional_rag(query, top_k, **kwargs)
                self.usage_stats['traditional_rag_used'] += 1
            
            execution_time = time.time() - start_time
            
            # 4. 构建结果
            result = L1AgentResult(
                success=True,
                answer=answer,
                method_used=selected_method,
                complexity_analysis=complexity_analysis,
                execution_time=execution_time,
                metadata={
                    'top_k': top_k,
                    'graph_rag_available': self.graph_rag_available,
                    'usage_stats': self.usage_stats.copy(),
                    **kwargs
                }
            )
            
            print(f"查询完成，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"L1 Agent查询失败: {str(e)}"
            print(error_msg)
            
            return L1AgentResult(
                success=False,
                answer="抱歉，查询处理过程中出现错误。",
                method_used="error",
                complexity_analysis=ComplexityAnalysisResult(
                    complexity_level='unknown',
                    confidence=0.0,
                    reasoning="查询处理异常",
                    recommended_method="traditional_rag",
                    features={}
                ),
                execution_time=execution_time,
                error_message=error_msg
            )