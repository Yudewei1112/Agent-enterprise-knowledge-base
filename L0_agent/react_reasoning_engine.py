"""ReAct推理引擎模块

该模块实现了ReAct (Reasoning and Acting) 推理框架，提供：
- 多跳推理规划和执行
- Thought-Action-Observation循环
- 推理链管理和依赖追踪
- 智能工具选择和参数优化
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI

# 统一使用绝对导入，避免类型检查问题
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from L0_agent_state import AgentState, ReActStep, ReasoningChain

from config import config
import logging


class ReActReasoningEngine:
    """ReAct推理引擎
    
    实现Thought-Action-Observation循环的多跳推理能力
    """
    
    def __init__(self, client: AsyncOpenAI = None):
        """初始化推理引擎
        
        参数:
            client: OpenAI客户端（可选，将根据模型动态创建）
        """
        self.client = client
        self._client_cache = {}  # 客户端缓存
        self.logger = logging.getLogger(__name__)
    
    def _get_or_create_client(self, model_name: str) -> AsyncOpenAI:
        """获取或创建客户端（带缓存）
        
        参数:
            model_name: 模型名称
            
        返回:
            OpenAI客户端实例
        """
        if model_name not in self._client_cache:
            model_config = config.get_model_config(model_name)
            if not model_config:
                raise ValueError(f"未找到模型配置: {model_name}")
            
            self._client_cache[model_name] = AsyncOpenAI(
                api_key=model_config["api_key"],
                base_url=model_config["api_base"],
                timeout=30.0,
                max_retries=3
            )
        return self._client_cache[model_name]
    
    async def analyze_intent_with_reasoning(self, query: str, context: List[str], 
                                          reasoning_chain: Optional[ReasoningChain] = None) -> Dict[str, Any]:
        """使用ReAct框架分析意图并规划推理
        
        参数:
            query: 用户查询
            context: 上下文信息
            reasoning_chain: 现有推理链
            
        返回:
            包含推理结果的字典
        """
        print(f"\n=== ReAct推理引擎: 意图分析 ===")
        
        # 首先评估查询复杂度
        complexity_assessment = await self._assess_query_complexity(query, context)
        print(f"🎯 复杂度评估: {complexity_assessment}")
        
        # 生成推理思考
        thought = await self._generate_react_thought(query, context, reasoning_chain)
        print(f"Thought: {thought}")
        
        # 规划多跳推理（已包含复杂度判断）
        reasoning_plan = await self._plan_multi_hop_reasoning(query, thought, context)
        print(f"推理规划: {reasoning_plan}")
        
        # 选择当前步骤的行动
        current_action = self._select_current_action(reasoning_plan, query)
        print(f"当前行动: {current_action}")
        
        return {
            'thoughts': [thought],
            'reasoning_chain': reasoning_chain,
            'reasoning_plan': reasoning_plan,
            'planned_actions': reasoning_plan,
            'current_action': current_action,
            'complexity_assessment': complexity_assessment  # 新增复杂度评估信息
        }
    
    async def _generate_react_thought(self, query: str, context: List[str], 
                                     reasoning_chain: Optional[ReasoningChain] = None) -> str:
        """生成ReAct思考过程
        
        参数:
            query: 用户查询
            context: 上下文信息
            reasoning_chain: 推理链
            
        返回:
            思考内容
        """
        # 构建思考提示词
        context_str = "\n".join(context[-3:]) if context else "无"
        
        previous_steps = ""
        if reasoning_chain and reasoning_chain.steps:
            steps_summary = []
            for step in reasoning_chain.steps[-2:]:  # 最近2步
                steps_summary.append(f"思考: {step.thought}\n行动: {step.action}\n观察: {step.observation}")
            previous_steps = "\n\n".join(steps_summary)
        
        prompt = f"""你是一个智能推理助手，需要分析用户问题并进行深度思考。

用户问题: {query}

上下文信息:
{context_str}

之前的推理步骤:
{previous_steps if previous_steps else "无"}

请进行深度思考，分析这个问题需要什么信息，应该采取什么策略。
思考要点:
1. 问题的核心是什么？
2. 需要哪些信息来回答？
3. 信息之间有什么关联？
4. 应该采用什么推理策略？

请提供你的思考过程（100-200字）:"""
        
        try:
            model_name = config.system_config['default_model']
            client = self._get_or_create_client(model_name)
            model_config = config.get_model_config(model_name)
            
            response = await client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个专业的推理分析师，擅长深度思考和逻辑分析。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"生成思考过程失败: {e}")
            return f"分析问题: {query}，需要收集相关信息进行回答。"
    
    async def _assess_query_complexity(self, query: str, context: List[str]) -> Dict[str, Any]:
        """评估查询复杂度
        
        参数:
            query: 用户查询
            context: 上下文信息
            
        返回:
            复杂度评估结果
        """
        prompt = f"""分析以下用户查询的复杂程度，并判断所需的推理策略。

用户查询: {query}
上下文: {context[:2] if context else ['无']}

请从以下维度评估:
1. **信息需求复杂度**: 是否需要多个信息源？
2. **推理步骤复杂度**: 是否需要多步逻辑推理？
3. **时间依赖性**: 是否涉及时间序列或因果关系？
4. **知识整合度**: 是否需要整合多领域知识？

复杂度分类:
- **简单**: 单一明确信息查询（如"公司地址"、"产品价格"、"员工手册内容"）
- **中等**: 需要简单推理或比较（如"最新产品特性对比"、"文档摘要"）
- **复杂**: 需要多步推理、时间关联或跨领域整合（如"比赛当天气温"、"多条件筛选"）

输出JSON格式:
{{
    "complexity_level": "simple|medium|complex",
    "reasoning_strategy": "direct|simplified|multi_hop",
    "estimated_steps": 1-4,
    "key_factors": ["因素1", "因素2"],
    "confidence": 0.0-1.0
}}"""
        
        try:
            model_name = config.system_config['default_model']
            client = self._get_or_create_client(model_name)
            model_config = config.get_model_config(model_name)
            
            response = await client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个查询复杂度分析专家，能够准确评估问题的推理难度。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析JSON
            try:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_content = result[json_start:json_end]
                    complexity_result = json.loads(json_content)
                    return complexity_result
            except json.JSONDecodeError:
                pass
            
            # 默认返回中等复杂度
            return {
                "complexity_level": "medium",
                "reasoning_strategy": "simplified",
                "estimated_steps": 2,
                "key_factors": ["默认评估"],
                "confidence": 0.5
            }
            
        except Exception as e:
            self.logger.error(f"复杂度评估失败: {e}")
            return {
                "complexity_level": "medium",
                "reasoning_strategy": "simplified",
                "estimated_steps": 2,
                "key_factors": ["评估失败"],
                "confidence": 0.3
            }
    
    async def _plan_simple_reasoning(self, query: str, thought: str, context: List[str]) -> List[Dict[str, Any]]:
        """简单问题推理规划：直接搜索 → 验证输出
        
        参数:
            query: 用户查询
            thought: 思考过程
            context: 上下文信息
            
        返回:
            简化的推理步骤列表
        """
        return [
            {
                "step_id": 1,
                "description": f"搜索查询相关信息: {query}",
                "tool": "local_document_rag_search",
                "reasoning": "直接搜索本地知识库获取答案",
                "dependencies": []
            },
            {
                "step_id": 2,
                "description": "验证信息准确性并生成答案",
                "tool": "verification",
                "reasoning": "确保信息准确并输出最终答案",
                "dependencies": [1]
            }
        ]
    
    async def _plan_simplified_reasoning(self, query: str, thought: str, context: List[str]) -> List[Dict[str, Any]]:
        """中等问题推理规划：本地搜索 → 补充搜索 → 整合输出
        
        参数:
            query: 用户查询
            thought: 思考过程
            context: 上下文信息
            
        返回:
            简化的推理步骤列表
        """
        return [
            {
                "step_id": 1,
                "description": f"获取基础信息: {query}",
                "tool": "local_document_rag_search",
                "reasoning": "首先从本地知识库获取基础信息",
                "dependencies": []
            },
            {
                "step_id": 2,
                "description": "补充或验证信息",
                "tool": "internet_search",
                "reasoning": "补充最新信息或验证本地信息",
                "dependencies": [1]
            },
            {
                "step_id": 3,
                "description": "整合信息并生成完整答案",
                "tool": "summary",
                "reasoning": "整合所有信息提供完整回答",
                "dependencies": [1, 2]
            }
        ]
    
    async def _plan_complex_reasoning(self, query: str, thought: str, context: List[str]) -> List[Dict[str, Any]]:
        """复杂问题推理规划：完整多跳推理
        
        参数:
            query: 用户查询
            thought: 思考过程
            context: 上下文信息
            
        返回:
            完整的推理步骤列表
        """
        # 保持原有的复杂推理逻辑
        return await self._plan_original_multi_hop_reasoning(query, thought, context)
    
    async def _plan_original_multi_hop_reasoning(self, query: str, thought: str, context: List[str]) -> List[Dict[str, Any]]:
        """原始的多跳推理规划逻辑
        
        参数:
            query: 用户查询
            thought: 思考过程
            context: 上下文信息
            
        返回:
            推理步骤规划列表
        """
        prompt = f"""基于以下信息，规划解决用户问题的推理步骤。

用户问题: {query}
思考分析: {thought}

可用工具:
1. local_document_rag_search - 搜索本地文档和知识库
2. internet_search - 联网搜索最新信息
3. mcp_service_search - 使用MCP服务检索

请规划2-4个推理步骤，每个步骤包含:
- step_id: 步骤编号
- description: 步骤描述（要具体明确，便于生成精确的搜索查询）
- tool: 使用的工具
- reasoning: 选择该工具的理由
- dependencies: 依赖的前置步骤（如果有）

重要要求:
1. 第一步和第二步的description应该明确要查找的具体信息（如日期、地点等）
2. 第三步和第四步应该是验证和总结步骤，不调用工具
3. 每个步骤都要有明确的目标，避免模糊描述
4. 工具选择规则：前两步使用搜索工具，后两步使用verification或summary

示例（针对"苏超南京VS苏州比赛当天的北京气温"）:
[
  {{
    "step_id": 1,
    "description": "确定苏超南京VS苏州比赛的具体日期",
    "tool": "internet_search",
    "reasoning": "需要先确定比赛日期才能查询当天气温",
    "dependencies": []
  }},
  {{
    "step_id": 2,
    "description": "查询比赛日期当天北京的气温数据",
    "tool": "internet_search",
    "reasoning": "根据第一步获得的日期查询具体气温",
    "dependencies": [1]
  }},
  {{
    "step_id": 3,
    "description": "验证比赛日期和气温信息的准确性",
    "tool": "verification",
    "reasoning": "确保获得的信息准确可靠",
    "dependencies": [1, 2]
  }},
  {{
    "step_id": 4,
    "description": "综合信息生成完整答案",
    "tool": "summary",
    "reasoning": "整合所有信息提供完整回答",
    "dependencies": [1, 2, 3]
  }}
]

输出JSON格式:
[
  {{
    "step_id": 1,
    "description": "步骤描述",
    "tool": "工具名称",
    "reasoning": "选择理由",
    "dependencies": []
  }}
]"""
        
        try:
            model_name = config.system_config['default_model']
            client = self._get_or_create_client(model_name)
            model_config = config.get_model_config(model_name)
            
            response = await client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个推理规划专家，能够将复杂问题分解为清晰的步骤。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析JSON
            try:
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_content = result[json_start:json_end]
                    plan = json.loads(json_content)
                    return plan
            except json.JSONDecodeError:
                pass
            
            # 如果解析失败，返回默认规划
            return [
                {
                    "step_id": 1,
                    "description": "搜索相关信息",
                    "tool": "local_document_rag_search",
                    "reasoning": "首先从本地知识库获取基础信息",
                    "dependencies": []
                }
            ]
            
        except Exception as e:
            self.logger.error(f"规划推理步骤失败: {e}")
            return [
                {
                    "step_id": 1,
                    "description": "搜索相关信息",
                    "tool": "local_document_rag_search",
                    "reasoning": "获取基础信息",
                    "dependencies": []
                }
            ]
    
    async def _plan_multi_hop_reasoning(self, query: str, thought: str, context: List[str]) -> List[Dict[str, Any]]:
        """规划多跳推理步骤 - 增强复杂度判断
        
        参数:
            query: 用户查询
            thought: 思考过程
            context: 上下文信息
            
        返回:
            推理步骤规划列表
        """
        # 1. 首先评估查询复杂度
        complexity_assessment = await self._assess_query_complexity(query, context)
        complexity_level = complexity_assessment.get('complexity_level', 'medium')
        reasoning_strategy = complexity_assessment.get('reasoning_strategy', 'simplified')
        
        print(f"🎯 复杂度评估: {complexity_level} | 策略: {reasoning_strategy}")
        
        # 2. 根据复杂度选择不同的推理路径
        if complexity_level == "simple" and reasoning_strategy == "direct":
            # 简单问题：直接路径
            plan = await self._plan_simple_reasoning(query, thought, context)
        elif complexity_level == "medium" and reasoning_strategy == "simplified":
            # 中等问题：简化路径
            plan = await self._plan_simplified_reasoning(query, thought, context)
        else:
            # 复杂问题：完整多跳推理
            plan = await self._plan_complex_reasoning(query, thought, context)
        
        # 3. 在计划中添加复杂度信息
        for step in plan:
            step['complexity_level'] = complexity_level
            step['reasoning_strategy'] = reasoning_strategy
        
        return plan
        prompt = f"""基于以下信息，规划解决用户问题的推理步骤。

用户问题: {query}
思考分析: {thought}

可用工具:
1. local_document_rag_search - 搜索本地文档和知识库
2. internet_search - 联网搜索最新信息
3. mcp_service_search - 使用MCP服务检索

请规划2-4个推理步骤，每个步骤包含:
- step_id: 步骤编号
- description: 步骤描述（要具体明确，便于生成精确的搜索查询）
- tool: 使用的工具
- reasoning: 选择该工具的理由
- dependencies: 依赖的前置步骤（如果有）

重要要求:
1. 第一步和第二步的description应该明确要查找的具体信息（如日期、地点等）
2. 第三步和第四步应该是验证和总结步骤，不调用工具
3. 每个步骤都要有明确的目标，避免模糊描述
4. 工具选择规则：前两步使用搜索工具，后两步使用verification或summary

示例（针对"苏超南京VS苏州比赛当天的北京气温"）:
[
  {{
    "step_id": 1,
    "description": "确定苏超南京VS苏州比赛的具体日期",
    "tool": "internet_search",
    "reasoning": "需要先确定比赛日期才能查询当天气温",
    "dependencies": []
  }},
  {{
    "step_id": 2,
    "description": "查询比赛日期当天北京的气温数据",
    "tool": "internet_search",
    "reasoning": "根据第一步获得的日期查询具体气温",
    "dependencies": [1]
  }},
  {{
    "step_id": 3,
    "description": "验证比赛日期和气温信息的准确性",
    "tool": "verification",
    "reasoning": "确保获得的信息准确可靠",
    "dependencies": [1, 2]
  }},
  {{
    "step_id": 4,
    "description": "综合信息生成完整答案",
    "tool": "summary",
    "reasoning": "整合所有信息提供完整回答",
    "dependencies": [1, 2, 3]
  }}
]

输出JSON格式:
[
  {{
    "step_id": 1,
    "description": "步骤描述",
    "tool": "工具名称",
    "reasoning": "选择理由",
    "dependencies": []
  }}
]"""
        
        try:
            model_name = config.system_config['default_model']
            client = self._get_or_create_client(model_name)
            model_config = config.get_model_config(model_name)
            
            response = await client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个推理规划专家，能够将复杂问题分解为清晰的步骤。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析JSON
            try:
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_content = result[json_start:json_end]
                    plan = json.loads(json_content)
                    return plan
            except json.JSONDecodeError:
                pass
            
            # 如果解析失败，返回默认规划
            return [
                {
                    "step_id": 1,
                    "description": "搜索相关信息",
                    "tool": "local_document_rag_search",
                    "reasoning": "首先从本地知识库获取基础信息",
                    "dependencies": []
                }
            ]
            
        except Exception as e:
            self.logger.error(f"规划推理步骤失败: {e}")
            return [
                {
                    "step_id": 1,
                    "description": "搜索相关信息",
                    "tool": "local_document_rag_search",
                    "reasoning": "获取基础信息",
                    "dependencies": []
                }
            ]
    
    def _select_current_action(self, reasoning_plan: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
        """选择当前应执行的行动
        
        参数:
            reasoning_plan: 推理规划
            query: 用户查询（用于生成参数）
            
        返回:
            当前行动
        """
        if not reasoning_plan:
            return {
                "tool": "local_document_rag_search",
                "description": "搜索相关信息",
                "parameters": {"query": query} if query else {"query": "搜索相关信息"}
            }
        
        # 选择第一个未完成的步骤
        current_step = reasoning_plan[0]
        tool_name = current_step.get("tool", "local_document_rag_search")
        
        # 根据工具类型生成合适的参数
        parameters = current_step.get("parameters", {})
        if not parameters:
            # 优先使用步骤描述，这样更精确
            step_query = current_step.get("description", query or "搜索相关信息")
            if tool_name == "local_document_rag_search":
                parameters = {"query": step_query}
            elif tool_name == "internet_search":
                parameters = {"query": step_query}
            elif tool_name == "mcp_service_search":
                parameters = {"query": step_query}
            else:
                parameters = {"query": step_query}
        
        return {
            "tool": tool_name,
            "description": current_step.get("description", "执行推理步骤"),
            "parameters": parameters,
            "reasoning": current_step.get("reasoning", "")
        }
    
    async def rewrite_query_for_next_step(self, original_query: str, step_description: str, 
                                         previous_results: str, step_index: int) -> str:
        """为下一步推理改写查询
        
        参数:
            original_query: 原始用户查询
            step_description: 当前步骤描述
            previous_results: 前面步骤的结果
            step_index: 当前步骤索引
            
        返回:
            改写后的查询
        """
        if step_index == 0:
            # 第一步，根据意图分析改写查询
            return await self._rewrite_first_step_query(original_query, step_description)
        else:
            # 后续步骤，结合前面的结果改写查询
            return await self._rewrite_subsequent_step_query(
                original_query, step_description, previous_results, step_index
            )
    
    async def _rewrite_first_step_query(self, original_query: str, step_description: str) -> str:
        """改写第一步查询
        
        参数:
            original_query: 原始查询
            step_description: 步骤描述
            
        返回:
            改写后的查询
        """
        prompt = f"""根据用户的原始问题和推理步骤描述，改写一个更精确的搜索查询。

原始问题: {original_query}
推理步骤: {step_description}

请生成一个简洁、精确的搜索查询，专注于这一步需要获取的信息。
只返回改写后的查询，不要其他内容。"""
        
        try:
            model_name = config.system_config['default_model']
            client = self._get_or_create_client(model_name)
            model_config = config.get_model_config(model_name)
            
            response = await client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个查询改写专家，能够根据推理步骤生成精确的搜索查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            return rewritten_query if rewritten_query else step_description
            
        except Exception as e:
            self.logger.error(f"改写第一步查询失败: {e}")
            return step_description
    
    async def _rewrite_subsequent_step_query(self, original_query: str, step_description: str, 
                                           previous_results: str, step_index: int) -> str:
        """改写后续步骤查询
        
        参数:
            original_query: 原始查询
            step_description: 步骤描述
            previous_results: 前面步骤的结果
            step_index: 步骤索引
            
        返回:
            改写后的查询
        """
        # 直接使用前面步骤的结果作为关键信息
        key_info = previous_results[:1000] if len(previous_results) > 1000 else previous_results
        
        prompt = f"""你需要根据前面步骤的结果改写一个精确的搜索查询。

【前面步骤的结果】:
{key_info}

【当前任务】: {step_description}

【关键要求】:
1. 仔细阅读前面步骤的结果，提取其中的关键信息（特别是日期、地点等）
2. 基于提取的信息生成搜索查询
3. 如果前面步骤提到了具体日期（如2025年7月5日），必须在查询中使用这个确切日期
4. 查询格式要简洁明确，如"2025年7月5日北京气温"

【示例】:
- 如果前面结果显示"比赛时间：2025年7月5日"，且当前任务是查询气温，则应生成"2025年7月5日北京气温"
- 如果前面结果显示"比赛时间：2024年3月15日"，且当前任务是查询天气，则应生成"2024年3月15日北京天气"

请严格按照上述要求，只返回改写后的查询，不要其他内容："""
        
        try:
            model_name = config.system_config['default_model']
            client = self._get_or_create_client(model_name)
            model_config = config.get_model_config(model_name)
            
            response = await client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个查询改写专家，能够结合前面步骤的结果生成精确的搜索查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            return rewritten_query if rewritten_query else step_description
            
        except Exception as e:
            self.logger.error(f"改写后续步骤查询失败: {e}")
            return step_description
    

    
    async def execute_reasoning_step(self, state: AgentState, action: Dict[str, Any], 
                                   tool_result: Any) -> ReActStep:
        """执行推理步骤并记录
        
        参数:
            state: 当前状态
            action: 执行的行动
            tool_result: 工具执行结果
            
        返回:
            推理步骤记录
        """
        # 生成观察结果
        observation = await self._generate_observation(action, tool_result, state)
        
        # 创建推理步骤
        step = ReActStep(
            step_id=state['react_step'] + 1,
            thought=state['thought_history'][-1] if state['thought_history'] else "执行推理步骤",
            action=f"{action['tool']}: {action['description']}",
            action_input=action.get('parameters', {}),
            observation=observation
        )
        
        # 更新推理链
        if state['reasoning_chain']:
            state['reasoning_chain'].steps.append(step)
        
        return step
    
    async def _generate_observation(self, action: Dict[str, Any], tool_result: Any, 
                                  state: AgentState) -> str:
        """生成观察结果
        
        参数:
            action: 执行的行动
            tool_result: 工具结果
            state: 当前状态
            
        返回:
            观察结果描述
        """
        if not tool_result or not hasattr(tool_result, 'success'):
            return "工具执行失败，未获得有效结果。"
        
        if not tool_result.success:
            return f"工具执行失败: {tool_result.error or '未知错误'}"
        
        # 返回实际的工具执行结果内容
        content = tool_result.content if tool_result.content else ""
        content_length = len(content)
        
        if content_length == 0:
            return "工具执行成功，但未返回任何内容。"
        elif content_length < 50:
            # 对于简短内容，直接返回完整内容
            return content
        elif content_length < 500:
            # 对于中等长度内容，返回完整内容
            return content
        else:
            # 对于长内容，返回前500个字符并添加省略号
            return content[:500] + "...(内容已截断)"
    
    async def comprehensive_reflection(self, state: AgentState) -> Dict[str, Any]:
        """综合反思评估
        
        参数:
            state: 当前状态
            
        返回:
            反思结果
        """
        print(f"\n=== ReAct推理引擎: 综合反思 ===")
        
        # 分析推理链完整性
        chain_analysis = self._analyze_reasoning_chain(state)
        
        # 评估答案质量
        answer_quality = await self._evaluate_answer_quality(state)
        
        # 确定下一步行动
        next_action = self._determine_next_action(state, chain_analysis, answer_quality)
        
        return {
            'chain_analysis': chain_analysis,
            'answer_quality': answer_quality,
            'next_action': next_action,
            'reflection_result': next_action['decision']
        }
    
    def _analyze_reasoning_chain(self, state: AgentState) -> Dict[str, Any]:
        """分析推理链完整性
        
        参数:
            state: 当前状态
            
        返回:
            推理链分析结果
        """
        chain = state.get('reasoning_chain')
        if not chain or not hasattr(chain, 'steps') or not chain.steps:
            return {
                'completeness': 0.0,
                'issues': ['推理链为空'],
                'suggestions': ['需要开始推理过程']
            }
        
        # 分析步骤完整性
        completed_steps = len([s for s in chain.steps if s.observation])
        total_planned = len(state.get('reasoning_plan', []))
        
        completeness = completed_steps / max(total_planned, 1)
        
        issues = []
        suggestions = []
        
        if completeness < 0.5:
            issues.append('推理步骤不完整')
            suggestions.append('需要继续执行推理步骤')
        
        if not state.get('retrieved_info'):
            issues.append('缺乏信息支撑')
            suggestions.append('需要获取更多信息')
        
        return {
            'completeness': completeness,
            'completed_steps': completed_steps,
            'total_planned': total_planned,
            'issues': issues,
            'suggestions': suggestions
        }
    
    async def _evaluate_answer_quality(self, state: AgentState) -> Dict[str, Any]:
        """评估答案质量
        
        参数:
            state: 当前状态
            
        返回:
            答案质量评估
        """
        if not state.get('current_answer'):
            return {
                'score': 0.0,
                'issues': ['没有生成答案'],
                'strengths': []
            }
        
        answer = state['current_answer']
        query = state['query']
        
        # 基础质量检查
        issues = []
        strengths = []
        
        if len(answer) < 50:
            issues.append('答案过于简短')
        else:
            strengths.append('答案长度适中')
        
        if '抱歉' in answer or '无法' in answer:
            issues.append('答案表达不确定性')
        else:
            strengths.append('答案表达确定')
        
        if state.get('retrieved_info'):
            strengths.append('基于检索信息生成')
        else:
            issues.append('缺乏信息支撑')
        
        # 计算质量分数
        score = max(0.0, min(1.0, len(strengths) / max(len(strengths) + len(issues), 1)))
        
        return {
            'score': score,
            'issues': issues,
            'strengths': strengths
        }
    
    def _determine_next_action(self, state: AgentState, chain_analysis: Dict[str, Any], 
                             answer_quality: Dict[str, Any]) -> Dict[str, Any]:
        """确定下一步行动
        
        参数:
            state: 当前状态
            chain_analysis: 推理链分析
            answer_quality: 答案质量评估
            
        返回:
            下一步行动决策
        """
        # 检查是否达到最大迭代次数
        if state.get('iteration_count', 0) >= 5:
            return {
                'decision': 'sufficient',
                'reason': '达到最大迭代次数',
                'action': 'finalize'
            }
        
        # 检查答案质量
        if answer_quality['score'] >= 0.7 and len(answer_quality['issues']) <= 1:
            return {
                'decision': 'sufficient',
                'reason': '答案质量良好',
                'action': 'finalize'
            }
        
        # 检查推理链完整性
        if chain_analysis['completeness'] < 0.8:
            return {
                'decision': 'insufficient',
                'reason': '推理链不完整，需要继续推理',
                'action': 'continue_reasoning'
            }
        
        # 检查信息充分性
        if not state.get('retrieved_info') or len(state.get('retrieved_info', '')) < 100:
            return {
                'decision': 'insufficient',
                'reason': '信息不足，需要更多检索',
                'action': 'gather_more_info'
            }
        
        # 默认继续推理
        return {
            'decision': 'insufficient',
            'reason': '需要进一步完善答案',
            'action': 'continue_reasoning'
        }