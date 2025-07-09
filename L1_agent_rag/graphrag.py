"""多图谱GraphRAG实现模块

该模块实现了基于多知识图谱的RAG系统，包括：
- 为每个文档单独构建知识图谱
- 实体和关系提取
- 社区检测和摘要生成
- 图谱路由：确定查询应该在哪些图谱中检索
- 并行检索：在选定的图谱中同时执行检索
- 结果融合：将多个图谱的结果进行智能合并
- 答案生成：基于融合结果生成最终答案
- 与现有LangGraph Agent的集成
"""

import os
import json
import asyncio
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import community as community_louvain
from openai import AsyncOpenAI

from .config import get_config
# from retrieval_methods import BaseRetriever  # 模块不存在，暂时注释

# 简单的BaseRetriever基类
class BaseRetriever:
    """检索器基类"""
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """检索方法"""
        raise NotImplementedError("子类必须实现retrieve方法")


@dataclass
class Entity:
    """实体类"""
    name: str
    type: str
    description: str = ""
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Relationship:
    """关系类"""
    source: str
    target: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Community:
    """社区类"""
    id: int
    entities: Set[str]
    relationships: List[Relationship]
    summary: str = ""
    level: int = 0
    

class GraphRAGExtractor:
    """GraphRAG实体关系提取器"""
    
    def __init__(self, client: AsyncOpenAI = None):
        """初始化提取器
        
        参数:
            client: OpenAI客户端实例
        """
        self.client = client or self._create_client()
        
    def _create_client(self) -> AsyncOpenAI:
        """创建OpenAI客户端"""
        config = get_config()
        default_model_config = config.get_model_config(config.default_model)
        return AsyncOpenAI(
            api_key=default_model_config["api_key"],
            base_url=default_model_config["api_base"]
        )
    
    async def extract_entities_and_relationships(self, text: str, max_entities: int = 20) -> Tuple[List[Entity], List[Relationship]]:
        """从文本中提取实体和关系
        
        参数:
            text: 输入文本
            max_entities: 最大实体数量
            
        返回:
            (实体列表, 关系列表)
        """
        print(f"开始提取实体和关系，文本长度: {len(text)}")
        prompt = f"""
你是一个专业的知识图谱构建专家。请从以下文本中提取实体和关系。

要求：
1. 提取最多{max_entities}个重要实体
2. 实体类型包括：人物、组织、地点、概念、事件、产品等
3. 关系要准确描述实体间的联系
4. 必须严格按照JSON格式输出，不要包含任何其他文本
5. 确保JSON格式正确，可以被解析

文本：
{text}

请严格按以下JSON格式输出，不要添加任何解释或其他内容：
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "实体类型",
      "description": "实体描述"
    }}
  ],
  "relationships": [
    {{
      "source": "源实体",
      "target": "目标实体",
      "relation_type": "关系类型",
      "description": "关系描述"
    }}
  ]
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=get_config().default_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱构建助手。请严格按照JSON格式输出，不要包含任何其他文本。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 尝试清理和修复JSON格式
            result_text = self._clean_json_response(result_text)
            
            # 解析JSON结果
            try:
                result = json.loads(result_text)
                entities = [Entity(**entity) for entity in result.get('entities', [])]
                relationships = [Relationship(**rel) for rel in result.get('relationships', [])]
                print(f"成功提取 {len(entities)} 个实体和 {len(relationships)} 个关系")
                return entities, relationships
            except json.JSONDecodeError as json_error:
                print(f"JSON解析失败: {str(json_error)}")
                print(f"原始输出：{result_text}")
                
                # 尝试使用正则表达式提取实体和关系
                entities, relationships = self._fallback_extraction(result_text)
                if entities or relationships:
                    print(f"回退提取成功: {len(entities)} 个实体和 {len(relationships)} 个关系")
                    return entities, relationships
                
                return [], []
                
        except Exception as e:
            print(f"实体关系提取失败: {str(e)}")
            return [], []
    
    def _clean_json_response(self, text: str) -> str:
        """清理和修复JSON响应
        
        参数:
            text: 原始响应文本
            
        返回:
            清理后的JSON文本
        """
        # 移除可能的markdown代码块标记
        text = text.replace('```json', '').replace('```', '')
        
        # 移除前后空白
        text = text.strip()
        
        # 查找JSON对象的开始和结束
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx]
        
        return text
    
    def _fallback_extraction(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """回退提取方法：使用正则表达式或简单解析
        
        参数:
            text: 原始响应文本
            
        返回:
            (实体列表, 关系列表)
        """
        entities = []
        relationships = []
        
        try:
            # 尝试提取实体信息
            import re
            
            # 查找实体模式
            entity_pattern = r'"name"\s*:\s*"([^"]+)".*?"type"\s*:\s*"([^"]+)".*?"description"\s*:\s*"([^"]+)"'
            entity_matches = re.findall(entity_pattern, text, re.DOTALL)
            
            for name, entity_type, description in entity_matches:
                entities.append(Entity(
                    name=name.strip(),
                    type=entity_type.strip(),
                    description=description.strip()
                ))
            
            # 查找关系模式
            rel_pattern = r'"source"\s*:\s*"([^"]+)".*?"target"\s*:\s*"([^"]+)".*?"relation_type"\s*:\s*"([^"]+)".*?"description"\s*:\s*"([^"]+)"'
            rel_matches = re.findall(rel_pattern, text, re.DOTALL)
            
            for source, target, relation_type, description in rel_matches:
                relationships.append(Relationship(
                    source=source.strip(),
                    target=target.strip(),
                    relation_type=relation_type.strip(),
                    description=description.strip()
                ))
            
        except Exception as e:
            print(f"回退提取失败: {str(e)}")
        
        return entities, relationships


class GraphRAGBuilder:
    """GraphRAG知识图谱构建器"""
    
    def __init__(self, extractor: GraphRAGExtractor = None):
        """初始化构建器
        
        参数:
            extractor: 实体关系提取器
        """
        self.extractor = extractor or GraphRAGExtractor()
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.communities: Dict[int, Community] = {}
        
    async def build_from_documents(self, documents: List[str]) -> None:
        """从文档列表构建知识图谱
        
        参数:
            documents: 文档内容列表
        """
        print(f"开始构建知识图谱，文档数量: {len(documents)}")
        
        all_entities = []
        all_relationships = []
        
        # 从每个文档提取实体和关系
        for i, doc in enumerate(documents):
            print(f"处理文档 {i+1}/{len(documents)}")
            entities, relationships = await self.extractor.extract_entities_and_relationships(doc)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # 合并重复实体
        self._merge_entities(all_entities)
        self.relationships = all_relationships
        
        # 构建NetworkX图
        self._build_networkx_graph()
        
        # 社区检测
        self._detect_communities()
        
        # 生成社区摘要
        await self._generate_community_summaries()
        
        print(f"知识图谱构建完成: {len(self.entities)}个实体, {len(self.relationships)}个关系, {len(self.communities)}个社区")
    
    def _merge_entities(self, entities: List[Entity]) -> None:
        """合并重复实体
        
        参数:
            entities: 实体列表
        """
        entity_map = {}
        
        for entity in entities:
            # 简单的实体合并策略：基于名称
            key = entity.name.lower().strip()
            if key in entity_map:
                # 合并描述
                existing = entity_map[key]
                if entity.description and entity.description not in existing.description:
                    existing.description += f"; {entity.description}"
                # 合并属性
                existing.attributes.update(entity.attributes)
            else:
                entity_map[key] = entity
        
        self.entities = {entity.name: entity for entity in entity_map.values()}
    
    def _build_networkx_graph(self) -> None:
        """构建NetworkX图"""
        # 添加节点
        for entity_name, entity in self.entities.items():
            self.graph.add_node(entity_name, **entity.attributes)
        
        # 添加边
        for rel in self.relationships:
            if rel.source in self.entities and rel.target in self.entities:
                self.graph.add_edge(
                    rel.source, 
                    rel.target, 
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight
                )
    
    def _detect_communities(self) -> None:
        """使用Louvain算法进行社区检测"""
        if len(self.graph.nodes) == 0:
            return
            
        # 使用Louvain算法
        partition = community_louvain.best_partition(self.graph)
        
        # 构建社区
        community_entities = defaultdict(set)
        for node, community_id in partition.items():
            community_entities[community_id].add(node)
        
        # 创建社区对象
        for community_id, entities in community_entities.items():
            # 获取社区内的关系
            community_relationships = [
                rel for rel in self.relationships
                if rel.source in entities and rel.target in entities
            ]
            
            self.communities[community_id] = Community(
                id=community_id,
                entities=entities,
                relationships=community_relationships
            )
    
    async def _generate_community_summaries(self) -> None:
        """生成社区摘要"""
        for community_id, community in self.communities.items():
            # 构建社区描述
            entities_desc = []
            for entity_name in community.entities:
                entity = self.entities.get(entity_name)
                if entity:
                    entities_desc.append(f"{entity.name}({entity.type}): {entity.description}")
            
            relationships_desc = []
            for rel in community.relationships:
                relationships_desc.append(f"{rel.source} -> {rel.relation_type} -> {rel.target}")
            
            # 生成摘要
            prompt = f"""
请为以下知识图谱社区生成一个简洁的摘要，描述这个社区的主要内容和特点：

实体：
{chr(10).join(entities_desc[:10])}  # 限制显示前10个实体

关系：
{chr(10).join(relationships_desc[:10])}  # 限制显示前10个关系

请用1-2句话总结这个社区的核心内容：
"""
            
            try:
                response = await self.extractor.client.chat.completions.create(
                    model=get_config().default_model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的知识图谱分析师。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                community.summary = response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"社区{community_id}摘要生成失败: {str(e)}")
                community.summary = f"包含{len(community.entities)}个实体的社区"
    
    def save_graph(self, filepath: str) -> None:
        """保存知识图谱到文件
        
        参数:
            filepath: 保存路径
        """
        graph_data = {
            'entities': {name: {
                'name': entity.name,
                'type': entity.type,
                'description': entity.description,
                'attributes': entity.attributes
            } for name, entity in self.entities.items()},
            'relationships': [{
                'source': rel.source,
                'target': rel.target,
                'relation_type': rel.relation_type,
                'description': rel.description,
                'weight': rel.weight,
                'attributes': rel.attributes
            } for rel in self.relationships],
            'communities': {str(cid): {
                'id': community.id,
                'entities': list(community.entities),
                'summary': community.summary,
                'level': community.level
            } for cid, community in self.communities.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"知识图谱已保存到: {filepath}")
    
    def load_graph(self, filepath: str) -> bool:
        """从文件加载知识图谱，增加错误处理
        
        参数:
            filepath: 知识图谱文件路径
            
        返回:
            bool: 加载成功返回True，失败返回False
        """
        try:
            if not os.path.exists(filepath):
                print(f"知识图谱文件不存在: {filepath}")
                return False
                
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 加载实体
            self.entities = {}
            for name, entity_data in graph_data['entities'].items():
                self.entities[name] = Entity(**entity_data)
            
            # 加载关系
            self.relationships = []
            for rel_data in graph_data['relationships']:
                self.relationships.append(Relationship(**rel_data))
            
            # 加载社区
            self.communities = {}
            for cid_str, community_data in graph_data['communities'].items():
                cid = int(cid_str)
                community_data['entities'] = set(community_data['entities'])
                # 重建关系列表
                community_relationships = [
                    rel for rel in self.relationships
                    if rel.source in community_data['entities'] and rel.target in community_data['entities']
                ]
                community_data['relationships'] = community_relationships
                self.communities[cid] = Community(**community_data)
            
            # 重建NetworkX图
            self._build_networkx_graph()
            
            print(f"知识图谱已加载: {len(self.entities)}个实体, {len(self.relationships)}个关系, {len(self.communities)}个社区")
            return True
            
        except Exception as e:
            print(f"加载知识图谱失败: {str(e)}")
            return False


class GraphRAGRetriever(BaseRetriever):
    """GraphRAG检索器"""
    
    def __init__(self, graph_builder: GraphRAGBuilder = None, embedding_model: SentenceTransformer = None):
        """初始化检索器
        
        参数:
            graph_builder: 知识图谱构建器
            embedding_model: 嵌入模型
        """
        self.graph_builder = graph_builder or GraphRAGBuilder()
        self.embedding_model = embedding_model or self._load_embedding_model()
        self.client = self.graph_builder.extractor.client
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """加载嵌入模型"""
        # 尝试加载本地模型，如果不存在则使用在线模型
        local_model_path = 'local_m3e_model'
        if os.path.exists(local_model_path):
            return SentenceTransformer(local_model_path)
        else:
            return SentenceTransformer('moka-ai/m3e-base')
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """执行GraphRAG检索
        
        参数:
            query: 查询文本
            **kwargs: 其他参数
                - search_type: 'global' 或 'local'
                - top_k: 返回结果数量
                
        返回:
            检索结果
        """
        search_type = kwargs.get('search_type', 'auto')
        top_k = kwargs.get('top_k', 5)
        
        # 自动判断搜索类型
        if search_type == 'auto':
            search_type = await self._determine_search_type(query)
        
        if search_type == 'global':
            return await self._global_search(query, top_k)
        else:
            return await self._local_search(query, top_k)
    
    async def _determine_search_type(self, query: str) -> str:
        """自动判断搜索类型
        
        参数:
            query: 查询文本
            
        返回:
            'global' 或 'local'
        """
        # 全局搜索关键词
        global_keywords = ['总结', '概述', '整体', '所有', '全部', '总体', '综合', '汇总']
        
        # 检查是否包含全局搜索关键词
        if any(keyword in query for keyword in global_keywords):
            return 'global'
        
        # 检查是否提及特定实体
        query_lower = query.lower()
        for entity_name in self.graph_builder.entities.keys():
            if entity_name.lower() in query_lower:
                return 'local'
        
        # 默认使用本地搜索
        return 'local'
    
    async def _global_search(self, query: str, top_k: int) -> str:
        """全局搜索：基于社区摘要
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            检索结果
        """
        if not self.graph_builder.communities:
            return "知识图谱中没有可用的社区信息。"
        
        # 计算查询与社区摘要的相似度
        query_embedding = self.embedding_model.encode(query)
        community_scores = []
        
        for community_id, community in self.graph_builder.communities.items():
            if community.summary:
                summary_embedding = self.embedding_model.encode(community.summary)
                similarity = np.dot(query_embedding, summary_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(summary_embedding)
                )
                community_scores.append((community_id, similarity, community))
        
        # 排序并选择top_k个社区
        community_scores.sort(key=lambda x: x[1], reverse=True)
        top_communities = community_scores[:top_k]
        
        # 构建上下文
        context_parts = []
        for community_id, score, community in top_communities:
            context_parts.append(f"社区{community_id}摘要: {community.summary}")
            
            # 添加关键实体信息
            key_entities = list(community.entities)[:5]  # 限制显示前5个实体
            for entity_name in key_entities:
                entity = self.graph_builder.entities.get(entity_name)
                if entity and entity.description:
                    context_parts.append(f"  - {entity.name}({entity.type}): {entity.description}")
        
        context = "\n".join(context_parts)
        
        # 生成最终答案
        prompt = f"""
基于以下知识图谱信息回答用户问题：

知识图谱信息：
{context}

用户问题：{query}

请基于提供的信息给出准确、全面的回答：
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=get_config().default_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"全局搜索失败: {str(e)}"
    
    async def _local_search(self, query: str, top_k: int) -> str:
        """本地搜索：基于实体邻居扩展
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            检索结果
        """
        # 提取查询中的实体
        query_entities = await self._extract_query_entities(query)
        
        if not query_entities:
            # 如果没有找到实体，使用向量相似度搜索
            return await self._similarity_search(query, top_k)
        
        # 扩展到邻居实体
        expanded_entities = set(query_entities)
        for entity in query_entities:
            if entity in self.graph_builder.graph:
                neighbors = list(self.graph_builder.graph.neighbors(entity))
                expanded_entities.update(neighbors[:3])  # 限制邻居数量
        
        # 构建上下文
        context_parts = []
        for entity_name in expanded_entities:
            entity = self.graph_builder.entities.get(entity_name)
            if entity:
                context_parts.append(f"{entity.name}({entity.type}): {entity.description}")
                
                # 添加相关关系
                related_rels = [
                    rel for rel in self.graph_builder.relationships
                    if rel.source == entity_name or rel.target == entity_name
                ]
                for rel in related_rels[:3]:  # 限制关系数量
                    context_parts.append(f"  关系: {rel.source} -> {rel.relation_type} -> {rel.target}")
        
        context = "\n".join(context_parts)
        
        # 生成最终答案
        prompt = f"""
基于以下知识图谱信息回答用户问题：

知识图谱信息：
{context}

用户问题：{query}

请基于提供的信息给出准确的回答：
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=get_config().default_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"本地搜索失败: {str(e)}"
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """从查询中提取实体
        
        参数:
            query: 查询文本
            
        返回:
            实体名称列表
        """
        found_entities = []
        query_lower = query.lower()
        
        # 简单的字符串匹配
        for entity_name in self.graph_builder.entities.keys():
            if entity_name.lower() in query_lower:
                found_entities.append(entity_name)
        
        return found_entities
    
    async def _similarity_search(self, query: str, top_k: int) -> str:
        """基于向量相似度的搜索
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            检索结果
        """
        query_embedding = self.embedding_model.encode(query)
        entity_scores = []
        
        for entity_name, entity in self.graph_builder.entities.items():
            if entity.description:
                entity_embedding = self.embedding_model.encode(entity.description)
                similarity = np.dot(query_embedding, entity_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
                )
                entity_scores.append((entity_name, similarity, entity))
        
        # 排序并选择top_k个实体
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        top_entities = entity_scores[:top_k]
        
        # 构建上下文
        context_parts = []
        for entity_name, score, entity in top_entities:
            context_parts.append(f"{entity.name}({entity.type}): {entity.description}")
        
        context = "\n".join(context_parts)
        
        # 生成最终答案
        prompt = f"""
基于以下知识图谱信息回答用户问题：

知识图谱信息：
{context}

用户问题：{query}

请基于提供的信息给出准确的回答：
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=get_config().default_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"相似度搜索失败: {str(e)}"


@dataclass
class GraphMetadata:
    """图谱元数据"""
    graph_id: str
    document_name: str
    document_path: str
    entities_summary: str = ""
    relationships_summary: str = ""
    topics: List[str] = None
    created_time: str = ""
    updated_time: str = ""
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []


class GraphRouter:
    """图谱路由器 - 使用向量检索方案确定查询应该在哪些图谱中检索"""
    
    def __init__(self, embedding_model: SentenceTransformer = None):
        """初始化路由器
        
        参数:
            embedding_model: 嵌入模型
        """
        self.embedding_model = embedding_model or self._load_embedding_model()
        self.graph_metadata: Dict[str, GraphMetadata] = {}
        self.last_routing_scores: Dict[str, float] = {}  # 用于调试
        
    def _load_embedding_model(self) -> SentenceTransformer:
        """加载嵌入模型"""
        # 尝试加载本地模型，如果不存在则使用在线模型
        local_model_path = 'local_m3e_model'
        if os.path.exists(local_model_path):
            return SentenceTransformer(local_model_path)
        else:
            return SentenceTransformer('moka-ai/m3e-base')
    
    def register_graph(self, graph_id: str, metadata: GraphMetadata) -> None:
        """注册图谱元数据
        
        参数:
            graph_id: 图谱ID
            metadata: 图谱元数据
        """
        self.graph_metadata[graph_id] = metadata
    
    def route_query(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """路由查询到相关图谱
        
        参数:
            query: 查询文本
            top_k: 返回的图谱数量
            
        返回:
            [(graph_id, score), ...] 按相关性排序
        """
        if not self.graph_metadata:
            return []
        
        # 计算查询向量
        query_embedding = self.embedding_model.encode(query)
        
        # 计算与每个图谱的相似度
        graph_scores = []
        
        for graph_id, metadata in self.graph_metadata.items():
            # 构建图谱描述文本
            graph_text_parts = []
            
            # 添加文档名
            graph_text_parts.append(f"文档：{metadata.document_name}")
            
            # 添加实体摘要
            if metadata.entities_summary:
                graph_text_parts.append(f"实体：{metadata.entities_summary}")
            
            # 添加关系摘要
            if metadata.relationships_summary:
                graph_text_parts.append(f"关系：{metadata.relationships_summary}")
            
            # 添加主题
            if metadata.topics:
                graph_text_parts.append(f"主题：{', '.join(metadata.topics)}")
            
            graph_text = " ".join(graph_text_parts)
            
            if graph_text.strip():
                # 计算相似度
                graph_embedding = self.embedding_model.encode(graph_text)
                similarity = np.dot(query_embedding, graph_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(graph_embedding)
                )
                graph_scores.append((graph_id, similarity))
        
        # 记录路由得分用于调试
        self.last_routing_scores = {graph_id: score for graph_id, score in graph_scores}
        
        # 排序并返回top_k
        graph_scores.sort(key=lambda x: x[1], reverse=True)
        return graph_scores[:top_k]


class MultiGraphRAGManager:
    """多图谱GraphRAG管理器"""
    
    def __init__(self):
        """初始化多图谱管理器"""
        # 加载配置
        config = get_config()
        
        # 修复路径构建逻辑：使用graph_storage_path而不是get_graph_path()
        self.base_graph_dir = os.path.join(config.graph_rag.graph_storage_path, 'multi_graphs')
        os.makedirs(self.base_graph_dir, exist_ok=True)
        
        # 图谱构建器字典 {graph_id: GraphRAGBuilder}
        self.graph_builders: Dict[str, GraphRAGBuilder] = {}
        
        # 图谱路由器
        self.router = GraphRouter()
        
        # 加载chunks目录路径
        self.chunks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chunks')
        
        # 加载已有图谱
        self._load_existing_graphs()
    
    def _load_existing_graphs(self) -> None:
        """加载已有的图谱"""
        if not os.path.exists(self.base_graph_dir):
            return
        
        for filename in os.listdir(self.base_graph_dir):
            if filename.endswith('.json'):
                graph_id = filename[:-5]  # 移除.json后缀
                graph_path = os.path.join(self.base_graph_dir, filename)
                
                # 创建图谱构建器并加载
                builder = GraphRAGBuilder()
                if builder.load_graph(graph_path):
                    self.graph_builders[graph_id] = builder
                    
                    # 生成并注册元数据
                    metadata = self._generate_graph_metadata(graph_id, builder)
                    self.router.register_graph(graph_id, metadata)
                    
                    print(f"已加载图谱: {graph_id}")
    
    def _generate_graph_metadata(self, graph_id: str, builder: GraphRAGBuilder) -> GraphMetadata:
        """生成图谱元数据
        
        参数:
            graph_id: 图谱ID
            builder: 图谱构建器
            
        返回:
            图谱元数据
        """
        # 提取实体类型统计
        entity_types = {}
        for entity in builder.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        entities_summary = ", ".join([f"{type_name}({count})" for type_name, count in entity_types.items()])
        
        # 提取关系类型统计
        relation_types = {}
        for rel in builder.relationships:
            relation_types[rel.relation_type] = relation_types.get(rel.relation_type, 0) + 1
        
        relationships_summary = ", ".join([f"{type_name}({count})" for type_name, count in relation_types.items()])
        
        # 提取主题（基于实体名称）
        topics = list(entity_types.keys())[:5]  # 取前5个实体类型作为主题
        
        return GraphMetadata(
            graph_id=graph_id,
            document_name=graph_id,  # 假设graph_id就是文档名
            document_path="",
            entities_summary=entities_summary,
            relationships_summary=relationships_summary,
            topics=topics
        )
    
    async def build_graphs_from_chunks(self, force_rebuild: bool = False) -> bool:
        """从chunks目录为每个文档构建独立的知识图谱
        
        参数:
            force_rebuild: 是否强制重建所有图谱
            
        返回:
            是否成功
        """
        if not os.path.exists(self.chunks_dir):
            print(f"Chunks目录不存在: {self.chunks_dir}")
            return False
        
        # 获取所有chunk文件
        chunk_files = [f for f in os.listdir(self.chunks_dir) if f.endswith('.json')]
        
        if not chunk_files:
            print("没有找到chunk文件")
            return False
        
        print(f"找到{len(chunk_files)}个文档的chunks，开始构建图谱...")
        
        success_count = 0
        for chunk_file in chunk_files:
            # 提取文档名作为graph_id
            graph_id = chunk_file[:-5]  # 移除.json后缀
            
            # 检查是否需要重建
            graph_path = os.path.join(self.base_graph_dir, f"{graph_id}.json")
            if os.path.exists(graph_path) and not force_rebuild:
                print(f"图谱已存在，跳过: {graph_id}")
                continue
            
            # 加载chunk文件
            chunk_file_path = os.path.join(self.chunks_dir, chunk_file)
            try:
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # 提取文档内容
                documents = []
                for chunk in chunk_data.get('chunks', []):
                    content = chunk.get('content', '')
                    if content:
                        documents.append(content)
                
                if not documents:
                    print(f"文档{graph_id}没有有效内容，跳过")
                    continue
                
                print(f"为文档{graph_id}构建知识图谱，包含{len(documents)}个chunks...")
                
                # 创建图谱构建器
                builder = GraphRAGBuilder()
                
                # 构建知识图谱
                await builder.build_from_documents(documents)
                
                # 检查构建结果
                if not builder.entities and not builder.relationships:
                    print(f"警告：文档{graph_id}构建的知识图谱为空，跳过保存")
                    continue
                
                print(f"准备保存图谱到: {graph_path}")
                print(f"图谱数据: {len(builder.entities)}个实体, {len(builder.relationships)}个关系, {len(builder.communities)}个社区")
                
                # 确保保存目录存在
                os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                
                # 保存图谱（增加详细异常处理）
                try:
                    builder.save_graph(graph_path)
                    
                    # 验证保存是否成功
                    if os.path.exists(graph_path):
                        # 检查文件大小
                        file_size = os.path.getsize(graph_path)
                        if file_size > 100:  # 至少100字节
                            print(f"图谱保存成功: {graph_path} (大小: {file_size} 字节)")
                        else:
                            print(f"警告：保存的图谱文件过小: {file_size} 字节")
                            # 读取文件内容检查
                            with open(graph_path, 'r', encoding='utf-8') as f:
                                saved_data = json.load(f)
                                print(f"保存的数据: entities={len(saved_data.get('entities', {}))}, relationships={len(saved_data.get('relationships', []))}")
                    else:
                        print(f"错误：图谱文件未成功创建: {graph_path}")
                        continue
                        
                except Exception as save_error:
                    print(f"保存图谱{graph_id}失败: {str(save_error)}")
                    print(f"保存路径: {graph_path}")
                    print(f"目录是否存在: {os.path.exists(os.path.dirname(graph_path))}")
                    print(f"目录权限: {os.access(os.path.dirname(graph_path), os.W_OK) if os.path.exists(os.path.dirname(graph_path)) else 'N/A'}")
                    import traceback
                    traceback.print_exc()  # 打印完整的错误堆栈
                    continue
                
                # 注册到管理器
                self.graph_builders[graph_id] = builder
                
                # 生成并注册元数据
                metadata = self._generate_graph_metadata(graph_id, builder)
                metadata.document_name = chunk_data.get('file_name', graph_id)
                metadata.document_path = chunk_data.get('file_path', '')
                self.router.register_graph(graph_id, metadata)
                
                success_count += 1
                print(f"图谱构建完成: {graph_id}")
                
            except Exception as e:
                print(f"构建图谱{graph_id}失败: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈
                continue
        
        print(f"图谱构建完成，成功构建{success_count}个图谱")
        return success_count > 0
    
    async def query(self, query: str, max_graphs: int = 3, top_k_per_graph: int = 5) -> str:
        """执行多图谱查询
        
        参数:
            query: 查询文本
            max_graphs: 最大查询图谱数量
            top_k_per_graph: 每个图谱返回的结果数量
            
        返回:
            查询结果
        """
        if not self.graph_builders:
            return "没有可用的知识图谱，请先构建图谱。"
        
        # 1. 图谱路由
        print(f"正在路由查询: {query}")
        selected_graphs = self.router.route_query(query, max_graphs)
        
        if not selected_graphs:
            return "没有找到相关的知识图谱。"
        
        print(f"选中{len(selected_graphs)}个图谱进行检索")
        
        # 2. 并行检索
        retrieval_tasks = []
        for graph_id, score in selected_graphs:
            if graph_id in self.graph_builders:
                builder = self.graph_builders[graph_id]
                retriever = GraphRAGRetriever(builder)
                task = self._retrieve_from_graph(retriever, query, graph_id, score, top_k_per_graph)
                retrieval_tasks.append(task)
        
        # 执行并行检索
        retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # 3. 结果融合
        valid_results = []
        for result in retrieval_results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                valid_results.append(result)
        
        if not valid_results:
            return "检索过程中发生错误，无法获取结果。"
        
        # 4. 答案生成
        return await self._generate_final_answer(query, valid_results)
    
    async def _retrieve_from_graph(self, retriever: GraphRAGRetriever, query: str, 
                                 graph_id: str, relevance_score: float, top_k: int) -> Dict[str, Any]:
        """从单个图谱检索
        
        参数:
            retriever: 检索器
            query: 查询文本
            graph_id: 图谱ID
            relevance_score: 相关性分数
            top_k: 返回结果数量
            
        返回:
            检索结果字典
        """
        try:
            # 自动判断搜索类型
            search_type = await retriever._determine_search_type(query)
            
            # 执行检索
            result = await retriever.retrieve(query, search_type=search_type, top_k=top_k)
            
            return {
                'graph_id': graph_id,
                'relevance_score': relevance_score,
                'search_type': search_type,
                'content': result,
                'metadata': self.router.graph_metadata.get(graph_id)
            }
            
        except Exception as e:
            print(f"从图谱{graph_id}检索失败: {str(e)}")
            return {
                'graph_id': graph_id,
                'relevance_score': relevance_score,
                'content': f"检索失败: {str(e)}",
                'error': True
            }
    
    async def _generate_final_answer(self, query: str, retrieval_results: List[Dict[str, Any]]) -> str:
        """生成最终答案
        
        参数:
            query: 原始查询
            retrieval_results: 检索结果列表
            
        返回:
            最终答案
        """
        # 构建上下文
        context_parts = []
        
        for i, result in enumerate(retrieval_results, 1):
            if result.get('error'):
                continue
                
            graph_id = result['graph_id']
            content = result['content']
            relevance_score = result['relevance_score']
            metadata = result.get('metadata')
            
            context_parts.append(f"\n=== 来源{i}: {metadata.document_name if metadata else graph_id} (相关性: {relevance_score:.3f}) ===")
            context_parts.append(content)
        
        if not context_parts:
            return "抱歉，没有找到相关信息。"
        
        context = "\n".join(context_parts)
        
        # 生成最终答案
        prompt = f"""
你是一个专业的知识图谱问答助手。请基于以下多个知识图谱的检索结果，为用户问题提供准确、全面的答案。

用户问题：{query}

检索结果：
{context}

请注意：
1. 综合多个来源的信息，提供完整的答案
2. 如果不同来源有冲突信息，请指出并分析
3. 明确标注信息来源
4. 如果信息不足，请诚实说明

请提供详细的回答：
"""
        
        try:
            # 创建客户端
            config = get_config()
            default_model_config = config.get_model_config(config.default_model)
            client = AsyncOpenAI(
                api_key=default_model_config["api_key"],
                base_url=default_model_config["api_base"]
            )
            
            response = await client.chat.completions.create(
                model=config.default_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱问答助手，擅长综合多个来源的信息提供准确答案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"答案生成失败: {str(e)}\n\n原始检索结果：\n{context}"
    
    def get_graphs_info(self) -> Dict[str, Any]:
        """获取所有图谱信息
        
        返回:
            图谱信息字典
        """
        graphs_info = {}
        
        for graph_id, builder in self.graph_builders.items():
            metadata = self.router.graph_metadata.get(graph_id)
            
            graphs_info[graph_id] = {
                'entities_count': len(builder.entities),
                'relationships_count': len(builder.relationships),
                'communities_count': len(builder.communities),
                'document_name': metadata.document_name if metadata else graph_id,
                'document_path': metadata.document_path if metadata else '',
                'entities_summary': metadata.entities_summary if metadata else '',
                'relationships_summary': metadata.relationships_summary if metadata else '',
                'topics': metadata.topics if metadata else []
            }
        
        return graphs_info
    
    def delete_graph(self, graph_id: str) -> bool:
        """删除指定图谱
        
        参数:
            graph_id: 图谱ID
            
        返回:
            是否成功删除
        """
        try:
            # 删除文件
            graph_path = os.path.join(self.base_graph_dir, f"{graph_id}.json")
            if os.path.exists(graph_path):
                os.remove(graph_path)
            
            # 从内存中移除
            if graph_id in self.graph_builders:
                del self.graph_builders[graph_id]
            
            if graph_id in self.router.graph_metadata:
                del self.router.graph_metadata[graph_id]
            
            print(f"图谱{graph_id}已删除")
            return True
            
        except Exception as e:
            print(f"删除图谱{graph_id}失败: {str(e)}")
            return False


class GraphRAGManager:
    """GraphRAG管理器（保持向后兼容）"""
    
    def __init__(self):
        """初始化GraphRAG管理器"""
        # 使用多图谱管理器
        self.multi_manager = MultiGraphRAGManager()
        
        # 保持向后兼容的属性
        self.graph_builder = None
        self.retriever = None
        
    async def initialize_from_documents(self, force_rebuild: bool = False) -> bool:
        """从chunks初始化多图谱GraphRAG（保持向后兼容）
        
        参数:
            force_rebuild: 是否强制重建
            
        返回:
            bool: 初始化成功返回True，失败返回False
        """
        return await self.multi_manager.build_graphs_from_chunks(force_rebuild)
    
    async def query(self, query: str, search_type: str = 'auto', top_k: int = 5) -> str:
        """执行多图谱GraphRAG查询（保持向后兼容）
        
        参数:
            query: 查询文本
            search_type: 搜索类型（在多图谱模式下忽略）
            top_k: 返回结果数量
            
        返回:
            查询结果
        """
        # 使用多图谱查询，max_graphs设为3，top_k_per_graph设为传入的top_k
        return await self.multi_manager.query(query, max_graphs=3, top_k_per_graph=top_k)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息（保持向后兼容）
        
        返回:
            统计信息字典
        """
        graphs_info = self.multi_manager.get_graphs_info()
        
        # 汇总统计信息
        total_entities = sum(info['entities_count'] for info in graphs_info.values())
        total_relationships = sum(info['relationships_count'] for info in graphs_info.values())
        total_communities = sum(info['communities_count'] for info in graphs_info.values())
        
        return {
            'graphs_count': len(graphs_info),
            'total_entities_count': total_entities,
            'total_relationships_count': total_relationships,
            'total_communities_count': total_communities,
            'graphs_info': graphs_info
        }
    
    # 新增方法：直接访问多图谱管理器
    def get_multi_manager(self) -> MultiGraphRAGManager:
        """获取多图谱管理器实例
        
        返回:
            多图谱管理器
        """
        return self.multi_manager


# 测试函数
async def test_graph_rag():
    """测试多图谱GraphRAG功能"""
    print("开始测试多图谱GraphRAG功能...")
    
    # 创建管理器
    manager = GraphRAGManager()
    
    # 初始化（从现有文档构建图谱）
    await manager.initialize_from_documents(force_rebuild=True)
    
    # 获取统计信息
    stats = manager.get_graph_stats()
    print(f"多图谱统计: {stats}")
    
    # 获取多图谱管理器进行更详细的测试
    multi_manager = manager.get_multi_manager()
    
    # 显示各个图谱的详细信息
    graphs_info = multi_manager.get_graphs_info()
    print(f"\n共有 {len(graphs_info)} 个独立图谱:")
    for graph_id, info in graphs_info.items():
        print(f"  - {graph_id}: {info['entities_count']}个实体, {info['relationships_count']}个关系, {info['communities_count']}个社区")
    
    # 测试查询（包含图谱路由、并行检索、结果融合）
    test_queries = [
        "请总结一下所有文档的主要内容",  # 全局搜索
        "亮化工程的预算是多少？",  # 本地搜索
        "有哪些公司参与了这些项目？",  # 实体搜索
        "不同文档中的项目有什么共同点？"  # 跨图谱查询测试
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("执行多图谱检索流程...")
        result = await manager.query(query)
        print(f"结果: {result}")
        print("-" * 50)


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_graph_rag())