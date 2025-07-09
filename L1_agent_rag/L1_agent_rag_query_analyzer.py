"""查询复杂度分析器

该模块负责分析用户查询的复杂度，决定使用传统RAG还是GraphRAG：
- 简单查询：直接事实查找、单一概念查询 -> 传统RAG
- 复杂查询：多跳推理、关系分析、综合总结 -> GraphRAG
"""

import re
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from .config import L1AgentConfig


@dataclass
class ComplexityAnalysisResult:
    """复杂度分析结果"""
    complexity_level: str  # 'simple' 或 'complex'
    confidence: float  # 置信度 0-1
    reasoning: str  # 分析理由
    recommended_method: str  # 'traditional_rag' 或 'graph_rag'
    features: Dict[str, bool]  # 检测到的特征


class QueryComplexityAnalyzer:
    """查询复杂度分析器"""
    
    def __init__(self, openai_client: AsyncOpenAI, config: L1AgentConfig = None):
        """初始化复杂度分析器
        
        Args:
            openai_client: OpenAI异步客户端
            config: L1Agent配置，如果为None则使用默认配置
        """
        self.client = openai_client
        self.l1_config = config or L1AgentConfig()
        self.config = self.l1_config.complexity_analysis
        
        # 缓存分析结果
        self.analysis_cache = {}
        self.cache_max_size = 100
        
        # 定义复杂查询的关键词和模式
        self.complex_keywords = self.config.complex_keywords
        self.simple_keywords = self.config.simple_keywords
        self.relationship_words = self.config.relationship_words
    

    
    async def analyze_complexity(self, query: str) -> ComplexityAnalysisResult:
        """分析查询复杂度
        
        参数:
            query: 用户查询
            
        返回:
            复杂度分析结果
        """
        # 1. 基于规则的初步分析
        rule_based_result = self._rule_based_analysis(query)
        
        # 2. 基于LLM的深度分析
        llm_based_result = await self._llm_based_analysis(query)
        
        # 3. 综合两种分析结果
        final_result = self._combine_results(rule_based_result, llm_based_result, query)
        
        return final_result
    
    def _rule_based_analysis(self, query: str) -> Dict:
        """基于规则的复杂度分析
        
        参数:
            query: 用户查询
            
        返回:
            分析结果字典
        """
        query_lower = query.lower()
        features = {}
        complexity_score = 0
        reasoning_parts = []
        
        # 检测复杂查询特征
        if isinstance(self.complex_keywords, dict):
            for category, keywords in self.complex_keywords.items():
                found_keywords = [kw for kw in keywords if kw in query_lower]
                if found_keywords:
                    features[f'has_{category}'] = True
                    complexity_score += len(found_keywords) * 2
                    reasoning_parts.append(f"包含{category}: {', '.join(found_keywords)}")
                else:
                    features[f'has_{category}'] = False
        else:
            # 兼容列表格式
            for keyword in self.complex_keywords:
                if keyword in query_lower:
                    features[f'has_complex_keyword_{keyword}'] = True
                    complexity_score += 2
                    reasoning_parts.append(f"包含复杂关键词: {keyword}")
        
        # 检测简单查询特征
        simple_score = 0
        if isinstance(self.simple_keywords, dict):
            for category, keywords in self.simple_keywords.items():
                found_keywords = [kw for kw in keywords if kw in query_lower]
                if found_keywords:
                    features[f'has_simple_{category}'] = True
                    simple_score += len(found_keywords) * 3
                    reasoning_parts.append(f"包含简单查询特征-{category}: {', '.join(found_keywords)}")
                else:
                    features[f'has_simple_{category}'] = False
        else:
            # 兼容列表格式
            for keyword in self.simple_keywords:
                if keyword in query_lower:
                    features[f'has_simple_keyword_{keyword}'] = True
                    simple_score += 3
                    reasoning_parts.append(f"包含简单关键词: {keyword}")
        
        # 其他复杂度指标
        query_length = len(query)
        word_count = len(query.split())
        question_marks = query.count('？') + query.count('?')
        
        features.update({
            'query_length': query_length,
            'word_count': word_count,
            'question_marks': question_marks,
            'has_multiple_questions': question_marks > 1
        })
        
        # 长查询通常更复杂
        if query_length > 50:
            complexity_score += 2
            reasoning_parts.append("查询较长")
        
        # 多个问号表示复杂查询
        if question_marks > 1:
            complexity_score += 3
            reasoning_parts.append("包含多个问题")
        
        # 计算最终复杂度
        if simple_score > complexity_score:
            complexity_level = 'simple'
            confidence = min(0.9, simple_score / (simple_score + complexity_score + 1))
        elif complexity_score >= 2:  # 降低复杂度阈值
            complexity_level = 'complex'
            confidence = min(0.9, complexity_score / (complexity_score + simple_score + 1))
        else:
            complexity_level = 'simple'  # 默认简单
            confidence = 0.5
        
        return {
            'complexity_level': complexity_level,
            'confidence': confidence,
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else '基于关键词分析',
            'features': features,
            'complexity_score': complexity_score,
            'simple_score': simple_score
        }
    
    async def _llm_based_analysis(self, query: str) -> Dict:
        """基于LLM的复杂度分析
        
        参数:
            query: 用户查询
            
        返回:
            分析结果字典
        """
        prompt = f"""
你是一个专业的查询复杂度分析专家。请分析以下用户查询的复杂度，并判断应该使用哪种检索方法。

查询: {query}

分析维度：
1. 是否需要多跳推理（需要通过多个实体和关系来回答）
2. 是否涉及实体间的复杂关系分析
3. 是否需要综合多个信息源
4. 是否需要对比、总结或深度分析

检索方法说明：
- 传统RAG：适用于直接事实查找、单一概念查询、简单问答
- GraphRAG：适用于关系分析、多跳推理、综合总结、复杂推理

请按以下JSON格式输出分析结果：
{{
  "complexity_level": "simple" 或 "complex",
  "confidence": 0.0-1.0的置信度,
  "reasoning": "详细的分析理由",
  "recommended_method": "traditional_rag" 或 "graph_rag",
  "requires_multi_hop": true/false,
  "requires_relationship_analysis": true/false,
  "requires_synthesis": true/false
}}
"""
        
        try:
            # 设置较短的超时时间，避免测试时等待过久
            import asyncio
            # 获取模型配置
            model_config = self.l1_config.get_model_config()
            
            # 检查API密钥是否有效
            api_key = model_config.get("api_key", "")
            if not api_key or api_key.startswith("your-") or api_key == "your-api-key":
                raise Exception(f"API密钥未正确配置。当前模型: {model_config.get('model', 'unknown')}，请设置正确的API密钥。")
            
            # 构建基础参数
            chat_params = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": "你是一个专业的查询复杂度分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": model_config.get("temperature", 0.1),
                "max_tokens": model_config.get("max_tokens", 4000)
            }
            
            # 为DeepSeek模型添加特殊参数
            if "deepseek" in model_config["model"].lower():
                chat_params["enable_thinking"] = False
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**chat_params),
                timeout=10.0  # 10秒超时
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON结果
            try:
                import json
                import re
                
                # 尝试提取JSON部分
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    result = json.loads(json_text)
                    print(f"✅ LLM分析JSON解析成功: {result}")
                    return result
                else:
                    print(f"❌ 无法从LLM响应中提取JSON: {result_text}")
                    raise json.JSONDecodeError("No JSON found", result_text, 0)
                    
            except json.JSONDecodeError as e:
                print(f"❌ LLM分析结果JSON解析失败: {str(e)}")
                print(f"原始响应: {result_text}")
                return {
                    'complexity_level': 'simple',
                    'confidence': 0.5,
                    'reasoning': 'LLM分析失败，使用默认判断',
                    'recommended_method': 'traditional_rag',
                    'requires_multi_hop': False,
                    'requires_relationship_analysis': False,
                    'requires_synthesis': False
                }
                
        except Exception as e:
            print(f"LLM复杂度分析失败: {str(e)}")
            return {
                'complexity_level': 'simple',
                'confidence': 0.5,
                'reasoning': f'LLM分析异常: {str(e)}',
                'recommended_method': 'traditional_rag',
                'requires_multi_hop': False,
                'requires_relationship_analysis': False,
                'requires_synthesis': False
            }
    
    def _combine_results(self, rule_result: Dict, llm_result: Dict, query: str) -> ComplexityAnalysisResult:
        """综合规则和LLM分析结果
        
        参数:
            rule_result: 规则分析结果
            llm_result: LLM分析结果
            query: 原始查询
            
        返回:
            最终分析结果
        """
        # 如果LLM分析失败（异常），完全依赖规则分析
        if 'LLM分析异常' in llm_result.get('reasoning', ''):
            final_complexity = rule_result['complexity_level']
            final_confidence = rule_result['confidence']
            reasoning = f"规则分析: {rule_result['reasoning']}; LLM分析: {llm_result['reasoning']}"
        else:
            # 权重设置：LLM分析权重更高
            rule_weight = 0.3
            llm_weight = 0.7
            
            # 复杂度判断
            rule_complex = rule_result['complexity_level'] == 'complex'
            llm_complex = llm_result['complexity_level'] == 'complex'
            
            # 综合置信度
            rule_confidence = rule_result['confidence']
            llm_confidence = llm_result['confidence']
            
            # 如果两者一致，置信度较高
            if rule_complex == llm_complex:
                final_complexity = rule_result['complexity_level']
                final_confidence = min(0.95, (rule_confidence * rule_weight + llm_confidence * llm_weight) * 1.2)
            else:
                # 如果不一致，以LLM结果为准，但降低置信度
                final_complexity = llm_result['complexity_level']
                final_confidence = max(0.3, llm_confidence * 0.8)
            
            reasoning = f"规则分析: {rule_result['reasoning']}; LLM分析: {llm_result['reasoning']}"
        
        # 推荐方法
        if final_complexity == 'complex':
            recommended_method = 'graph_rag'
        else:
            recommended_method = 'traditional_rag'
        
        # 合并特征
        combined_features = rule_result['features'].copy()
        combined_features.update({
            'llm_requires_multi_hop': llm_result.get('requires_multi_hop', False),
            'llm_requires_relationship_analysis': llm_result.get('requires_relationship_analysis', False),
            'llm_requires_synthesis': llm_result.get('requires_synthesis', False)
        })
        
        # 合并推理过程
        combined_reasoning = f"规则分析: {rule_result['reasoning']}; LLM分析: {llm_result['reasoning']}"
        
        return ComplexityAnalysisResult(
            complexity_level=final_complexity,
            confidence=final_confidence,
            reasoning=combined_reasoning,
            recommended_method=recommended_method,
            features=combined_features
        )
    
    def get_analysis_summary(self, result: ComplexityAnalysisResult) -> str:
        """获取分析摘要
        
        参数:
            result: 分析结果
            
        返回:
            分析摘要字符串
        """
        method_name = "GraphRAG" if result.recommended_method == 'graph_rag' else "传统RAG"
        complexity_name = "复杂" if result.complexity_level == 'complex' else "简单"
        
        return f"查询复杂度: {complexity_name} (置信度: {result.confidence:.2f}) | 推荐方法: {method_name}"