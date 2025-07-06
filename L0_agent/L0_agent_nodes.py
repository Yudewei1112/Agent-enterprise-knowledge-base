"""Agent节点模块

该模块定义了LangGraph中的各个节点，包括：
- 意图分析与路由节点
- 工具执行节点
- 反思评估节点
- 答案生成节点
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from openai import AsyncOpenAI

# 统一使用绝对导入，避免类型检查问题
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from L0_agent_state import AgentState, ToolResult, cleanup_temporary_state
from agent_state_validator import safe_update_state, validate_state_integrity, log_state_issues, validate_cleaned_state
from L0_agent_tools import AgentToolManager
from react_reasoning_engine import ReActReasoningEngine
from config import config
from database import db
import logging


class AgentNodes:
    """Agent节点集合"""
    
    def __init__(self, tool_manager: AgentToolManager, react_reasoning_engine: ReActReasoningEngine):
        """初始化节点
        
        参数:
            tool_manager: 工具管理器
            react_reasoning_engine: ReAct推理引擎实例
        """
        self.tool_manager = tool_manager
        self.react_reasoning_engine = react_reasoning_engine
        self.client = None
        self._client_cache = {}  # 客户端缓存，避免重复创建
    
    def _create_client_for_model(self, model_name: str) -> AsyncOpenAI:
        """为指定模型创建客户端
        
        参数:
            model_name: 模型名称
            
        返回:
            OpenAI客户端实例
        """
        model_config = config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"未找到模型配置: {model_name}")
        
        return AsyncOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["api_base"],
            timeout=30.0,
            max_retries=3
        )
    
    def _get_or_create_client(self, model_name: str) -> AsyncOpenAI:
        """获取或创建客户端（带缓存）
        
        参数:
            model_name: 模型名称
            
        返回:
            OpenAI客户端实例
        """
        if model_name not in self._client_cache:
            self._client_cache[model_name] = self._create_client_for_model(model_name)
        return self._client_cache[model_name]
    
    async def _enhance_tool_parameters(self, tool_name: str, parameters: dict, state: AgentState) -> dict:
        """为工具增强参数（优化版本）
        
        使用改写后的查询替代历史会话传递，提高工具查询质量
        
        参数:
            tool_name: 工具名称
            parameters: 原始参数字典
            state: 当前状态
            
        返回:
            增强后的参数字典
        """
        enhanced_params = parameters.copy()
        
        # 获取工具实例以检查其参数需求
        tool = self.tool_manager.get_tool(tool_name)
        if not tool:
            print(f"警告: 工具 {tool_name} 不存在，使用原始参数")
            return enhanced_params
            
        if not hasattr(tool, 'args_schema'):
            print(f"警告: 工具 {tool_name} 没有args_schema，使用原始参数")
            return enhanced_params
        
        # 节点(analyze_intent_node/continue_react_reasoning_node)已经将改写后的查询
        # 放入了parameters中，这里不再重复处理，以避免状态不同步问题。
        if 'query' in enhanced_params:
            print(f"使用节点准备的查询: {enhanced_params.get('query', 'N/A')}")
        
        # 检查工具是否需要模型名称参数
        if (hasattr(tool, 'args_schema') and 
            tool.args_schema is not None and 
            hasattr(tool.args_schema, '__fields__') and 
            tool.args_schema.__fields__ is not None and
            'model_name' in tool.args_schema.__fields__):
            enhanced_params['model_name'] = state.get('model_name') or config.system_config['default_model']
        
        # 可以在这里添加更多通用参数的处理逻辑
        # 例如：用户ID、会话ID、时间戳等
        
        return enhanced_params
    
    async def analyze_intent_node(self, state: AgentState) -> AgentState:
        """意图分析节点 - 增强版ReAct推理
        
        功能:
        - 分析用户查询意图
        - 生成ReAct推理步骤
        - 选择合适的工具
        - 设置工具参数
        - 管理推理链状态
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态
        """
        print(f"\n=== ReAct增强意图分析节点 ===")
        print(f"用户查询: {state['query']}")
        print(f"已使用工具: {state['used_tools']}")
        print(f"迭代次数: {state['iteration_count']}")
        print(f"当前状态键: {list(state.keys())}")
        
        # 获取可用工具描述
        available_tools_desc = self.tool_manager.get_tool_descriptions(state['used_tools'])
        
        if not self.tool_manager.get_available_tools(state['used_tools']):
            # 没有可用工具了
            state['messages'].append("所有工具都已使用，准备生成最终答案")
            return state
        
        try:
            # 使用ReAct推理引擎进行意图分析和推理规划
            context = [state.get('retrieved_info', '')] if state.get('retrieved_info') else []
            reasoning_result = await self.react_reasoning_engine.analyze_intent_with_reasoning(
                state['query'], 
                context,
                state.get('reasoning_chain')
            )
            
            # 获取复杂度评估结果
            complexity_info = reasoning_result.get('complexity_assessment', {})
            complexity_level = complexity_info.get('complexity_level', 'medium')
            reasoning_strategy = complexity_info.get('reasoning_strategy', 'simplified')
            complexity_confidence = complexity_info.get('confidence', 0.5)
            complexity_factors = complexity_info.get('key_factors', [])
            
            print(f"🎯 问题复杂度: {complexity_level} | 策略: {reasoning_strategy} | 置信度: {complexity_confidence}")
            
            # 记录复杂度信息到状态
            state['query_complexity'] = complexity_level
            state['reasoning_strategy'] = reasoning_strategy
            state['complexity_confidence'] = complexity_confidence
            state['complexity_factors'] = complexity_factors
            
            # 获取推理计划
            reasoning_plan = reasoning_result.get('planned_actions', [])
            print(f"📋 ReAct推理计划: {len(reasoning_plan)}个步骤 (复杂度: {complexity_level})")
            
            # 如果有推理计划，执行多步骤推理
            if reasoning_plan and len(reasoning_plan) > 1:
                print(f"🔄 开始执行多步骤ReAct推理 ({len(reasoning_plan)}步)")
                
                # 初始化推理链（如果还没有）
                if not state.get('reasoning_chain'):
                    from L0_agent_state import ReasoningChain
                    state['reasoning_chain'] = ReasoningChain()
                
                # 存储推理计划到状态
                state['reasoning_plan'] = reasoning_plan
                state['current_step_index'] = 0
                
                # 执行第一个步骤
                current_step = reasoning_plan[0]
                current_action = {
                    'tool': current_step.get('tool', 'local_document_rag_search'),
                    'description': current_step.get('description', '执行推理步骤'),
                    'parameters': current_step.get('parameters', {'query': state['query']})
                }
                
                # 确保工具名称有效
                tool_name = current_action.get('tool', 'local_document_rag_search')
                if not tool_name or tool_name is None or tool_name == '':
                    tool_name = 'local_document_rag_search'
                    print(f"⚠️ 工具名称无效，使用默认工具: {tool_name}")
                
                # 改写第一步查询
                step_description = current_step.get('description', '执行推理步骤')
                rewritten_query = await self.react_reasoning_engine.rewrite_query_for_next_step(
                    state['query'], step_description, "", 0  # 第一步，没有前面的结果
                )
                
                # 更新工具参数中的查询
                updated_parameters = current_action['parameters'].copy()
                updated_parameters['query'] = rewritten_query
                
                # 更新状态
                react_updates = {
                    'selected_tool': tool_name,
                    'tool_parameters': updated_parameters,
                    'rewritten_query': rewritten_query,
                    'current_reasoning_goal': current_step.get('description', '执行推理步骤'),
                    'reasoning_confidence': 0.8,
                    'multi_step_reasoning': True  # 标记为多步骤推理
                }
                
                print(f"📍 执行第1/{len(reasoning_plan)}步: {current_step.get('description')}")
                print(f"🔧 使用工具: {current_action['tool']}")
                print(f"📝 工具参数: {current_action['parameters']}")
                
            else:
                # 单步推理或没有计划
                current_action = reasoning_result.get('current_action', {})
                react_updates = {
                    'selected_tool': current_action.get('tool', 'local_document_rag_search'),
                    'tool_parameters': current_action.get('parameters', {}),
                    'rewritten_query': state['query'],
                    'current_reasoning_goal': '分析用户意图并选择合适工具',
                    'reasoning_plan': reasoning_plan,
                    'reasoning_confidence': 0.8,
                    'multi_step_reasoning': False
                }
                
                print(f"📍 执行单步推理")
                print(f"🔧 使用工具: {current_action.get('tool', 'N/A')}")
            
            # 更新推理历史
            if reasoning_result.get('thoughts'):
                from datetime import datetime
                for thought in reasoning_result['thoughts']:
                    state['thought_history'].append({
                        'step': len(state['thought_history']) + 1,
                        'thought': thought,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.8
                    })
            
            # 安全更新状态
            update_result = safe_update_state(state, react_updates)
            if not all(update_result.values()):
                print("ReAct意图分析节点状态更新警告")
                log_state_issues(state)
            
            # 记录详细的推理信息到消息历史
            thoughts_str = '; '.join(reasoning_result.get('thoughts', ['已完成思考']))
            state['messages'].append(f"🧠 ReAct推理思考: {thoughts_str}")
            state['messages'].append(f"🎯 问题复杂度: {state.get('query_complexity', 'medium')} | 推理策略: {state.get('reasoning_strategy', 'simplified')}")
            state['messages'].append(f"📋 推理计划: {len(reasoning_plan)}个步骤 (复杂度驱动)")
            state['messages'].append(f"🔧 选择工具: {state.get('selected_tool', 'N/A')}")
            state['messages'].append(f"📊 复杂度置信度: {state.get('complexity_confidence', 0.5):.2f} | 推理置信度: 0.8")
            if state.get('complexity_factors'):
                factors_str = ', '.join(state['complexity_factors'])
                state['messages'].append(f"🔍 复杂度因素: {factors_str}")
            
            print(f"✅ ReAct推理完成:")
            print(f"   思考: {thoughts_str}")
            print(f"   目标: {state.get('current_reasoning_goal', '分析用户意图')}")
            print(f"   计划: {len(reasoning_plan)}个步骤")
            print(f"   工具: {state.get('selected_tool', 'N/A')}")
            print(f"   参数: {state.get('tool_parameters', {})}")
            print(f"   置信度: 0.8")
            
        except Exception as e:
            error_msg = f"ReAct意图分析失败: {str(e)}"
            state['messages'].append(error_msg)
            print(f"ReAct意图分析异常: {e}")
            
            # 回退到传统意图分析方法
            print("回退到传统意图分析方法...")
            
            # 准备格式化变量
            used_tools_str = ', '.join(state['used_tools']) if state['used_tools'] else '无'
            retrieved_info_str = state['retrieved_info'] if state['retrieved_info'] else '无'
            available_tools_desc = self.tool_manager.get_tool_descriptions(state['used_tools'])
            
            # 构建意图分析提示词 - 避免复杂的f-string嵌套
            user_query = state['query']
            
            # 获取对话历史用于查询改写
            conversation_history = []
            if state.get('conversation_id'):
                try:
                    messages = await db.get_messages(state['conversation_id'])
                    conversation_history = [
                        {
                            'role': 'user' if msg[2] == 'user' else 'assistant',
                            'content': msg[1]
                        }
                        for msg in messages[-5:]  # 最近5条消息
                    ]
                except Exception as e:
                    print(f"获取对话历史失败: {str(e)}")
            
            # 构建对话历史字符串
            history_str = "无"
            if conversation_history:
                history_parts = []
                for msg in conversation_history:
                    role_name = "用户" if msg['role'] == 'user' else "助手"
                    history_parts.append(f"{role_name}: {msg['content']}")
                history_str = "\n".join(history_parts)
            
            prompt_parts = [
                "你是一个智能企业助理的决策核心。你的任务是：1）分析用户的提问并结合历史对话改写查询；2）从一系列可用工具中选择最合适的一个来获取信息。",
                "",
                f"用户当前查询: {user_query}",
                "",
                "历史对话:",
                history_str,
                "",
                "可用工具:",
                available_tools_desc,
                "",
                f"已使用的工具: {used_tools_str}",
                "",
                "当前已获取的信息:",
                retrieved_info_str,
                "",
                "**# 指令:**",
                "1.  仔细阅读用户的当前查询和历史对话。",
                "2.  **查询改写:** 基于历史对话上下文，将用户的当前查询改写为一个完整、独立的查询，包含必要的上下文信息。",
                "3.  **工具选择:** 回顾可用工具列表，理解每个工具的功能、适用场景和局限性。",
                "4.  **关键决策:** 基于改写后的查询，决定哪个工具最有可能找到回答用户问题的关键信息。",
                "5.  **思考过程:** 请在 <reasoning> 标签中简要陈述你的决策理由。",
                "6.  **输出格式:** 你的最终决策必须以一个 JSON 对象的形式提供，包含 `rewritten_query`、`tool_name`和`parameters`。",
                "",
                "**# 上下文信息:**",
                f"* **用户当前问题:** {user_query}",
                f"* **历史对话:** {history_str}",
                f"* **可用工具列表:** {available_tools_desc}",
                f"* **已使用的工具:** {used_tools_str}",
                "    *注意：你必须且只能从可用工具列表中选择一个工具，且不能选择已使用的工具。*",
                "",
                "**# 开始分析:**",
                "<reasoning>",
                "... 在这里进行你的思考 ...",
                "</reasoning>",
                "",
                "{",
                '  "rewritten_query": "改写后的完整查询",',
                '  "tool_name": "selected_tool",',
                '  "parameters": {',
                '    "query": "改写后的查询内容"',
                '  }',
                "}",
                "",
                "#### **示例:**",
                "",
                "假设用户问：我们公司最新的AI服务器产品规格是什么？和市面上英伟达最新的产品比怎么样？",
                "并且，这是第一次查询，所有工具都可用。",
                "",
                "**模型理想的输出:**",
                "",
                "<reasoning>",
                "用户的问题包含两个部分：1. 公司内部的AI服务器规格。2. 与市面上英伟达最新产品的对比。",
                "第一部分明显是内部信息，local_document_rag_search 工具的描述（产品手册、技术规格）与此完美匹配，应优先使用它来获取内部产品信息。",
                "第二部分虽然需要联网，但首先我需要知道我们自己的产品是什么，才能进行比较。因此，第一步是调用内部检索工具。",
                "</reasoning>",
                "{",
                '  "tool_name": "local_document_rag_search",',
                '  "parameters": {',
                '    "query": "最新AI服务器产品规格"',
                '  }',
                "}",
                "",
                "注意: 对于local_document_rag_search工具，如果需要指定文件，请使用specific_file参数，不要使用document_name。"
            ]
            
            prompt = "\n".join(prompt_parts)
            
            try:
                model_name = state.get('model_name') or config.system_config['default_model']
                model_config = config.get_model_config(model_name)
                if not model_config:
                    raise ValueError(f"模型配置错误: {model_name}")
                
                # 为当前模型获取或创建客户端（使用缓存）
                current_client = self._get_or_create_client(model_name)
                
                # 构建基础参数
                base_params = {
                    "model": model_config["model"],
                    "messages": [
                        {"role": "system", "content": "你是一个专业的意图分析助手，能够准确分析用户查询并选择合适的工具。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
                
                # 使用统一的参数获取方法
                call_params = config.get_model_call_params(model_name, base_params)
                
                try:
                    response = await current_client.chat.completions.create(**call_params)
                except Exception as e:
                    print(f"模型调用失败 - 模型: {model_name}, 错误: {str(e)}")
                    print(f"调用参数: {call_params}")
                    # 回退到基础参数重试
                    basic_params = {k: v for k, v in call_params.items() 
                                   if k in ['model', 'messages', 'temperature', 'max_tokens']}
                    response = await current_client.chat.completions.create(**basic_params)
                
                analysis_result = response.choices[0].message.content.strip()
                print(f"意图分析结果: {analysis_result}")
                
                # 解析JSON响应 - 增强容错性
                try:
                    # 尝试提取JSON部分
                    json_start = analysis_result.find('{')
                    json_end = analysis_result.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_content = analysis_result[json_start:json_end]
                        print(f"提取的JSON内容: {json_content}")
                        
                        analysis_data = json.loads(json_content)
                        rewritten_query = analysis_data.get('rewritten_query', state['query'])  # 获取改写后的查询
                        selected_tool = analysis_data.get('tool_name')  # 修正字段名从selected_tool到tool_name
                        reasoning = analysis_data.get('reasoning', '')
                        parameters = analysis_data.get('parameters', {})
                        
                        # 调试：检查第一步查询改写条件
                        print(f"🔍 调试 - multi_step_reasoning: {state.get('multi_step_reasoning', False)}")
                        print(f"🔍 调试 - current_step_index: {state.get('current_step_index', 0)}")
                        print(f"🔍 调试 - reasoning_plan长度: {len(state.get('reasoning_plan', []))}")
                        
                        # 如果是多步骤推理的第一步，使用ReAct引擎改写查询
                        if state.get('multi_step_reasoning', False) and state.get('current_step_index', 0) == 0:
                            print(f"✅ 进入第一步查询改写逻辑")
                            reasoning_plan = state.get('reasoning_plan', [])
                            if reasoning_plan:
                                first_step = reasoning_plan[0]
                                print(f"🔍 第一步描述: {first_step.get('description', '')}")
                                try:
                                    # 直接调用第一步查询改写方法
                                    first_step_rewritten_query = await self.react_reasoning_engine._rewrite_first_step_query(
                                        state['query'],  # 原始查询
                                        first_step.get('description', '')  # 步骤描述
                                    )
                                    print(f"🔄 第一步改写后的查询: {first_step_rewritten_query}")
                                    
                                    # 重要：更新改写后的查询变量
                                    rewritten_query = first_step_rewritten_query
                                    
                                    # 更新参数中的查询
                                    if 'query' in parameters:
                                        parameters['query'] = first_step_rewritten_query
                                        print(f"✅ 已更新parameters中的query: {parameters['query']}")
                                except Exception as e:
                                    print(f"第一步查询改写失败: {e}")
                                    # 如果改写失败，使用步骤描述作为查询
                                    rewritten_query = first_step.get('description', state['query'])
                                    if 'query' in parameters:
                                        parameters['query'] = rewritten_query
                            else:
                                print(f"❌ reasoning_plan为空，无法进行第一步查询改写")
                        else:
                            print(f"❌ 不满足第一步查询改写条件")
                        
                        # 确保工具名称有效
                        if not selected_tool or selected_tool is None or selected_tool == '':
                            selected_tool = 'local_document_rag_search'
                            print(f"⚠️ 解析的工具名称无效，使用默认工具: {selected_tool}")
                        
                        print(f"解析结果 - 改写查询: {rewritten_query}")
                        print(f"解析结果 - 工具: {selected_tool}, 参数: {parameters}")
                        
                        if selected_tool and selected_tool in self.tool_manager.get_available_tools(state['used_tools']):
                            state['messages'].append(f"改写查询: {rewritten_query}")
                            state['messages'].append(f"选择工具: {selected_tool}")
                            state['messages'].append(f"选择理由: {reasoning}")
                            
                            # 存储改写后的查询、选择的工具和参数，供下一个节点使用
                            update_results = safe_update_state(state, {
                                'rewritten_query': rewritten_query,
                                'selected_tool': selected_tool,
                                'tool_parameters': parameters
                            })
                            
                            if not all(update_results.values()):
                                print(f"状态更新警告: {update_results}")
                                log_state_issues(state)
                            
                            # 状态更新后的调试输出
                            print(f"是否有selected_tool: {'selected_tool' in state}")
                            print(f"成功选择工具: {selected_tool}")
                            print(f"工具参数: {parameters}")
                        else:
                            available_tools = list(self.tool_manager.get_available_tools(state['used_tools']).keys())
                            state['messages'].append(f"工具选择失败: {selected_tool} 不在可用工具列表中: {available_tools}")
                            print(f"工具选择失败: {selected_tool} 不在可用工具列表中: {available_tools}")
                    else:
                        state['messages'].append(f"无法从响应中提取有效JSON: {analysis_result}")
                        print(f"无法提取JSON，原始响应: {analysis_result}")
                        
                except json.JSONDecodeError as e:
                    state['messages'].append(f"意图分析结果解析失败: {str(e)}")
                    print(f"JSON解析错误: {e}")
                    print(f"原始响应内容: {repr(analysis_result)}")
                    
            except Exception as e:
                error_msg = f"意图分析失败: {str(e)}"
                state['messages'].append(error_msg)
                print(f"意图分析异常: {e}")
        
        return state
    
    async def continue_react_reasoning_node(self, state: AgentState) -> AgentState:
        """继续ReAct多步骤推理节点
        
        功能:
        - 检查是否需要继续执行推理步骤
        - 执行下一个推理步骤
        - 更新推理链状态
        - 决定是否完成推理
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态
        """
        print(f"\n=== ReAct多步骤推理继续节点 ===")
        
        # 检查是否是多步骤推理
        if not state.get('multi_step_reasoning', False):
            print("非多步骤推理，跳过")
            return state
        
        reasoning_plan = state.get('reasoning_plan', [])
        current_step_index = state.get('current_step_index', 0)
        
        if not reasoning_plan:
            print("推理计划无效")
            state['multi_step_reasoning'] = False
            return state
        
        print(f"📍 当前步骤索引: {current_step_index + 1}, 总步骤数: {len(reasoning_plan)}")
        
        # 检查是否有工具执行结果，如果有则说明当前步骤已完成，需要准备下一步
        if state.get('last_tool_result') is not None:
            print(f"✅ 第{current_step_index + 1}步已完成，准备下一步")
            
            # 记录当前步骤的执行结果
            try:
                current_step = reasoning_plan[current_step_index]
                current_action = {
                    'tool': current_step.get('tool', 'unknown'),
                    'description': current_step.get('description', '执行推理步骤'),
                    'parameters': current_step.get('parameters', {})
                }
                
                # 使用ReAct引擎记录推理步骤
                react_step = await self.react_reasoning_engine.execute_reasoning_step(
                    state, current_action, state['last_tool_result']
                )
                
                print(f"✅ 记录推理步骤: {react_step.action}")
                print(f"📝 观察结果: {react_step.observation[:100]}...")
            except Exception as e:
                print(f"记录推理步骤失败: {e}")
            
            # 递增步骤索引，准备下一步
            current_step_index += 1
            state['current_step_index'] = current_step_index
        # 如果没有工具执行结果，说明是首次进入，直接执行当前步骤
        # 不需要特殊处理，因为analyze_intent_node只是设置了第一步的工具和参数
        
        # 检查是否还有更多步骤需要执行
        if current_step_index < len(reasoning_plan):
            # 获取当前要执行的步骤
            current_step = reasoning_plan[current_step_index]
            
            print(f"📍 准备执行第{current_step_index + 1}/{len(reasoning_plan)}步")
            print(f"📋 步骤描述: {current_step.get('description', '未知步骤')}")
            
            # 设置当前步骤的工具和参数
            current_action = {
                'tool': current_step.get('tool', 'local_document_rag_search'),
                'description': current_step.get('description', '执行推理步骤'),
                'parameters': current_step.get('parameters', {'query': state['query']})
            }
            
            # 确保工具名称有效
            tool_name = current_action.get('tool', 'local_document_rag_search')
            if not tool_name or tool_name is None or tool_name == '':
                tool_name = 'local_document_rag_search'
                print(f"⚠️ 工具名称无效，使用默认工具: {tool_name}")
            
            # 为当前步骤改写查询（基于前面步骤的结果）
            try:
                # 使用前面步骤生成的答案作为上下文信息
                previous_results = state.get('current_answer', '') or str(state.get('last_tool_result', ''))
                step_rewritten_query = await self.react_reasoning_engine.rewrite_query_for_next_step(
                    state['query'],
                    current_step.get('description', '执行推理步骤'),
                    previous_results,
                    current_step_index
                )
                print(f"🔄 第{current_step_index + 1}步查询改写: {step_rewritten_query}")
                
            except Exception as e:
                print(f"查询改写失败，使用原始查询: {e}")
                step_rewritten_query = current_action['parameters'].get('query', state['query'])
            
            # 更新工具参数中的查询
            updated_parameters = current_action['parameters'].copy()
            updated_parameters['query'] = step_rewritten_query
            
            # 记录当前步骤的改写查询到step_wise_results（预备记录）
            if current_step_index > 0:  # 第二步及以后才记录
                current_step_info = {
                    'step_number': current_step_index + 1,
                    'rewritten_query': step_rewritten_query,
                    'selected_tool': tool_name,
                    'tool_parameters': updated_parameters,
                    'preliminary_answer': '',  # Will be updated in tool_executor_node
                    'confidence': 0.8,
                    'observation': '',
                    'timestamp': datetime.now().isoformat()
                }
                
                # 如果step_wise_results中还没有这一步的记录，则添加
                if len(state['step_wise_results']) < current_step_index + 1:
                    state['step_wise_results'].append(current_step_info)
                else:
                    # 更新已有记录的改写查询
                    state['step_wise_results'][current_step_index]['rewritten_query'] = step_rewritten_query
                    state['step_wise_results'][current_step_index]['selected_tool'] = tool_name
            
            # 更新状态为当前步骤
            react_updates = {
                'selected_tool': tool_name,
                'tool_parameters': updated_parameters,
                'rewritten_query': step_rewritten_query,
                'current_reasoning_goal': current_step.get('description', '执行推理步骤'),
                'last_tool_result': None,  # 清除上一步结果
                'tool_execution_status': None  # 清除工具执行状态，准备当前步骤
            }
            
            update_result = safe_update_state(state, react_updates)
            if not all(update_result.values()):
                print("ReAct继续推理节点状态更新警告")
                log_state_issues(state)
            
            print(f"🔧 当前步骤工具: {react_updates['selected_tool']}")
            print(f"📝 工具参数: {react_updates['tool_parameters']}")
            print(f"🎯 步骤目标: {react_updates['current_reasoning_goal']}")
            
            # 记录到消息历史
            state['messages'].append(f"🔄 执行ReAct推理第{current_step_index + 1}/{len(reasoning_plan)}步")
            state['messages'].append(f"🎯 当前目标: {current_step.get('description')}")
            state['messages'].append(f"🔧 使用工具: {tool_name}")
            
        else:
            # 所有步骤已完成
            print("🎉 所有推理步骤已完成")
            state['multi_step_reasoning'] = False
            state['messages'].append("🎉 ReAct多步骤推理已完成")
            
            # 清除工具选择状态，确保路由到反思节点
            safe_update_state(state, {
                'selected_tool': None,
                'tool_parameters': {},
                'tool_execution_status': 'completed'
            })
            
            # 进行最终反思
            try:
                reflection_result = await self.react_reasoning_engine.comprehensive_reflection(state)
                print(f"🤔 推理反思: {reflection_result.get('reflection_result', '完成')}")
                state['messages'].append(f"🤔 推理反思: {reflection_result.get('reflection_result', '完成')}")
            except Exception as e:
                print(f"推理反思失败: {e}")
        
        return state


    
    async def tool_executor_node(self, state: AgentState) -> AgentState:
        """工具执行节点 - 增强版ReAct推理
        
        功能:
        - 执行选定的工具
        - 记录ReAct推理步骤
        - 处理工具执行结果
        - 生成观察结果
        - 更新推理链状态
        - 累积检索信息
        - 生成初步答案
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态
        """
        print(f"\n=== ReAct增强工具执行节点 ===")
        print(f"当前状态键: {list(state.keys())}")
        print(f"是否有selected_tool: {'selected_tool' in state}")
        
        selected_tool = state.get('selected_tool')
        tool_parameters = state.get('tool_parameters', {})
        
        print(f"selected_tool值: {selected_tool}")
        print(f"selected_tool类型: {type(selected_tool)}")
        print(f"tool_parameters值: {tool_parameters}")
        
        # 严格检查selected_tool
        if not selected_tool or selected_tool is None or selected_tool == '':
            print("没有选择的工具，跳过执行")
            state['messages'].append("❌ 没有选择的工具，跳过执行")
            # 标记工具执行状态为失败，避免无限循环
            safe_update_state(state, {'tool_execution_status': 'no_tool_selected'})
            return state
            
        # 检查工具是否存在
        if not self.tool_manager.get_tool(selected_tool):
            print(f"工具 {selected_tool} 不存在")
            state['messages'].append(f"❌ 工具 {selected_tool} 不存在")
            safe_update_state(state, {'tool_execution_status': 'tool_not_found'})
            return state
        
        print(f"🔧 开始执行工具: {selected_tool}")
        print(f"📋 工具参数: {tool_parameters}")
        
        # 记录ReAct推理步骤 - Action阶段
        action_description = f"使用{selected_tool}工具执行查询"
        state['action_history'].append({
            'step': len(state['action_history']) + 1,
            'action': selected_tool,
            'action_input': tool_parameters,
            'description': action_description,
            'timestamp': datetime.now().isoformat()
        })
        
        # 为工具添加通用参数（解耦合具体工具名称）
        tool_parameters = await self._enhance_tool_parameters(selected_tool, tool_parameters, state)
        
        try:
            # 执行工具
            result = await self.tool_manager.execute_tool_async(selected_tool, **tool_parameters)
            
            # 检查工具执行结果是否成功且有有效内容
            tool_success = self._is_tool_result_valid(result, selected_tool)
            
            if tool_success:
                print(f"✅ 工具执行成功")
                state['messages'].append(f"✅ 工具 {selected_tool} 执行成功")
                
                # 使用ReAct推理引擎执行推理步骤并生成观察结果
                try:
                    # 构造action字典
                    action = {
                        'tool': selected_tool,
                        'description': f'执行工具 {selected_tool}',
                        'parameters': tool_parameters
                    }
                    
                    reasoning_step_result = await self.react_reasoning_engine.execute_reasoning_step(
                        state,
                        action,
                        result
                    )
                    
                    # 更新推理链状态
                    if state['reasoning_chain']:
                        state['reasoning_chain'].add_step(
                            thought=reasoning_step_result.thought,
                            action=selected_tool,
                            action_input=tool_parameters,
                            observation=reasoning_step_result.observation
                        )
                    
                    # 记录观察结果到历史
                    state['observation_history'].append({
                        'step': len(state['observation_history']) + 1,
                        'tool': selected_tool,
                        'observation': reasoning_step_result.observation,
                        'confidence': reasoning_step_result.confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # 更新已使用工具
                    state['used_tools'].add(selected_tool)
                    
                    # 累积检索信息
                    new_retrieved_info = (
                        state['retrieved_info'] + f"\n\n=== {selected_tool} 检索结果 ===\n{result.content}"
                        if state['retrieved_info'] 
                        else f"=== {selected_tool} 检索结果 ===\n{result.content}"
                    )
                    
                    # 先更新retrieved_info，然后生成答案
                    safe_update_state(state, {'retrieved_info': new_retrieved_info})
                    
                    # 生成基于当前信息的初步答案
                    answer = await self._generate_preliminary_answer(state)
                    
                    # 更新或记录当前步骤的详细信息到step_wise_results
                    current_step_index = state.get('current_step_index', 0)
                    
                    # 检查是否已经有预记录的步骤信息（来自continue_react_reasoning_node）
                    if (current_step_index < len(state['step_wise_results']) and 
                        state['step_wise_results'][current_step_index].get('preliminary_answer') == ''):
                        # 更新现有记录的preliminary_answer和其他执行结果
                        state['step_wise_results'][current_step_index].update({
                            'preliminary_answer': answer,
                            'confidence': reasoning_step_result.confidence,
                            'observation': reasoning_step_result.observation,
                            'tool_parameters': tool_parameters,  # 更新实际使用的参数
                            'timestamp': datetime.now().isoformat()  # 更新执行时间
                        })
                        print(f"📝 更新第{current_step_index + 1}步的执行结果")
                    else:
                        # 添加新的步骤记录
                        current_step_info = {
                            'step_number': len(state['step_wise_results']) + 1,
                            'rewritten_query': state.get('rewritten_query', state['query']),
                            'selected_tool': selected_tool,
                            'tool_parameters': tool_parameters,
                            'preliminary_answer': answer,
                            'confidence': reasoning_step_result.confidence,
                            'observation': reasoning_step_result.observation,
                            'timestamp': datetime.now().isoformat()
                        }
                        state['step_wise_results'].append(current_step_info)
                        print(f"📝 添加第{len(state['step_wise_results'])}步的执行结果")
                    
                    # 处理多步骤推理状态更新
                    state_updates = {
                        'current_answer': answer,
                        'tool_execution_status': 'success',  # 标记工具执行成功
                        'react_step': state['react_step'] + 1  # 增加ReAct步骤计数
                    }
                    
                    # 设置工具执行结果，供continue_react_reasoning_node使用
                    state_updates['last_tool_result'] = result
                    
                    # 检查是否是多步骤推理
                    if state.get('multi_step_reasoning', False):
                        reasoning_plan = state.get('reasoning_plan', [])
                        current_step_index = state.get('current_step_index', 0)
                        
                        print(f"🔄 多步骤推理: 完成第{current_step_index + 1}/{len(reasoning_plan)}步")
                        
                        # 检查是否还有下一步
                        if current_step_index + 1 < len(reasoning_plan):
                            next_step = reasoning_plan[current_step_index + 1]
                            print(f"📋 准备执行下一步: {next_step.get('description', '未知步骤')}")
                            
                            # 检查步骤类型，第三步和第四步通常是验证和总结，不需要工具调用
                            step_tool = next_step.get('tool', '')
                            if step_tool in ['verification', 'summary', 'analysis']:
                                print(f"🔍 第{current_step_index + 2}步为验证/总结步骤，不调用工具")
                                # 直接进行验证或总结，不调用工具
                                state_updates['multi_step_reasoning'] = False
                                state_updates['react_step_is_final'] = True  # 标记为最后一步
                                state_updates['selected_tool'] = None  # 清除工具选择
                                state_updates['tool_parameters'] = {}  # 清除工具参数
                            else:
                                # 继续使用工具，标记工具执行完成以便continue_react_reasoning_node处理
                                state_updates['react_step_is_final'] = False  # 标记不是最后一步
                                state_updates['selected_tool'] = None  # 清除当前工具选择，让continue_react_reasoning_node重新设置
                                state_updates['tool_parameters'] = {}  # 清除当前工具参数
                                print(f"🔄 继续多步推理，下一步将由continue_react_reasoning_node处理")
                        else:
                            print(f"🎉 多步骤推理所有步骤完成")
                            state_updates['multi_step_reasoning'] = False
                            state_updates['react_step_is_final'] = True  # 标记为最后一步
                            state_updates['selected_tool'] = None  # 清除工具选择
                            state_updates['tool_parameters'] = {}  # 清除工具参数
                    else:
                        # 非多步推理，标记为最后一步
                        state_updates['react_step_is_final'] = True
                    
                    # 安全更新状态
                    update_results = safe_update_state(state, state_updates)
                    
                    if not all(update_results.values()):
                        print(f"ReAct工具执行节点状态更新警告: {update_results}")
                        log_state_issues(state)
                    
                    # 记录详细的ReAct推理信息
                    observation_preview = str(reasoning_step_result.observation)[:200] if reasoning_step_result.observation else '无观察结果'
                    answer_preview = str(answer)[:100] if answer else '无答案'
                    state['messages'].append(f"🔍 观察结果: {observation_preview}...")
                    state['messages'].append(f"📊 推理置信度: {reasoning_step_result.confidence:.2f}")
                    
                    print(f"🔍 ReAct观察结果: {observation_preview[:100]}...")
                    print(f"📊 推理置信度: {reasoning_step_result.confidence:.2f}")
                    print(f"💡 生成初步答案: {answer_preview}...")
                    
                except Exception as react_error:
                    print(f"⚠️ ReAct推理步骤执行失败，回退到传统方式: {react_error}")
                    state['messages'].append(f"⚠️ ReAct推理失败，使用传统方式处理")
                    
                    # 回退到传统处理方式
                    state['used_tools'].add(selected_tool)
                    new_retrieved_info = (
                        state['retrieved_info'] + f"\n\n=== {selected_tool} 检索结果 ===\n{result.content}"
                        if state['retrieved_info'] 
                        else f"=== {selected_tool} 检索结果 ===\n{result.content}"
                    )
                    safe_update_state(state, {'retrieved_info': new_retrieved_info})
                    answer = await self._generate_preliminary_answer(state)
                    safe_update_state(state, {
                        'current_answer': answer,
                        'tool_execution_status': 'success'
                    })
                    answer_preview = str(answer)[:100] if answer else '无答案'
                    print(f"💡 生成初步答案: {answer_preview}...")
                
            else:
                # 工具执行失败或返回无效内容
                failure_reason = "执行失败" if not result.success else "返回内容为空或无效"
                print(f"工具执行失败: {failure_reason} - {result.error if result.error else '无错误信息'}")
                state['messages'].append(f"工具 {selected_tool} {failure_reason}: {result.error if result.error else '返回内容无效'}")
                
                # 更新重试计数
                if selected_tool not in state['tool_retry_counts']:
                    state['tool_retry_counts'][selected_tool] = 0
                state['tool_retry_counts'][selected_tool] += 1
                
                # 标记工具执行失败
                safe_update_state(state, {'tool_execution_status': 'failed'})
                
                # 检查是否达到重试上限（3次）
                if state['tool_retry_counts'][selected_tool] >= 3:
                    print(f"工具 {selected_tool} 失败次数达到3次，准备直接输出结果")
                    state['messages'].append(f"工具 {selected_tool} 重试3次均失败，将基于现有信息生成答案")
                    
                    # 生成基于现有信息的答案
                    if state['retrieved_info']:
                        answer = await self._generate_preliminary_answer(state)
                    else:
                        answer = f"抱歉，我无法通过工具检索找到相关信息来回答您的问题: {state['query']}。请尝试重新表述您的问题或提供更多上下文信息。"
                    
                    safe_update_state(state, {
                        'current_answer': answer,
                        'tool_execution_status': 'max_retries_reached'
                    })
                else:
                    # 还可以重试，但不更新used_tools，允许重新选择同一工具
                    print(f"工具 {selected_tool} 失败，当前重试次数: {state['tool_retry_counts'][selected_tool]}/3")
                
        except Exception as e:
            error_msg = f"工具执行异常: {str(e)}"
            print(f"工具执行异常: {e}")
            state['messages'].append(error_msg)
            
            # 更新重试计数
            if selected_tool not in state['tool_retry_counts']:
                state['tool_retry_counts'][selected_tool] = 0
            state['tool_retry_counts'][selected_tool] += 1
            
            # 标记工具执行异常
            safe_update_state(state, {'tool_execution_status': 'exception'})
            
            # 检查是否达到重试上限
            if state['tool_retry_counts'][selected_tool] >= 3:
                print(f"工具 {selected_tool} 异常次数达到3次，准备直接输出结果")
                state['messages'].append(f"工具 {selected_tool} 异常3次，将基于现有信息生成答案")
                
                # 生成基于现有信息的答案
                if state['retrieved_info']:
                    answer = await self._generate_preliminary_answer(state)
                else:
                    answer = f"抱歉，在处理您的问题时遇到技术问题: {state['query']}。请稍后重试或联系技术支持。"
                
                safe_update_state(state, {
                    'current_answer': answer,
                    'tool_execution_status': 'max_retries_reached'
                })
        
        return state
    
    def _is_tool_result_valid(self, result, tool_name: str) -> bool:
        """判断工具执行结果是否有效
        
        参数:
            result: 工具执行结果
            tool_name: 工具名称
            
        返回:
            bool: 结果是否有效
        """
        if not result.success or not result.content or not result.content.strip():
            return False
        
        content_lower = result.content.lower()
        
        # 通用无效内容检查
        if content_lower in ['none', 'null', 'empty', '无', '空']:
            return False
        
        # MCP工具特定的失败信息检查
        mcp_failure_indicators = [
            "当前没有可用的MCP工具",
            "没有找到可用的MCP工具", 
            "no available mcp tools",
            "mcp工具不可用",
            "mcp connection failed"
        ]
        
        for indicator in mcp_failure_indicators:
            if indicator.lower() in content_lower:
                return False
        
        # 可以根据需要添加其他工具的特定失败检查
        
        return True
    

    
    async def _generate_preliminary_answer(self, state: AgentState) -> str:
        """生成基于当前检索信息的初步答案
        
        参数:
            state: 当前状态
            
        返回:
            生成的初步答案
        """
        prompt = f"""基于以下检索到的信息，为用户问题生成一个准确、完整的答案。

用户问题: {state.get('rewritten_query') or state['query']}

检索到的信息:
{state['retrieved_info']}

请生成一个清晰、准确的答案。如果信息不足以完全回答问题，请说明需要更多信息。
需要特别注意的是，如果信息来源是本地知识库检索，如果召回的信息是不匹配的，那么不要强行使用该信息回答，直接回答未能检索到正确信息"""
        
        try:
            model_name = state.get('model_name') or config.system_config['default_model']
            model_config = config.get_model_config(model_name)
            if not model_config:
                return f"模型配置错误: {model_name}，无法生成答案。"
            
            # 为当前模型获取或创建客户端（使用缓存）
            current_client = self._get_or_create_client(model_name)
            
            # 构建基础参数
            base_params = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": "你是一个智能助手，能够基于提供的信息准确回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            # 使用统一的参数获取方法
            call_params = config.get_model_call_params(model_name, base_params)
            
            try:
                response = await current_client.chat.completions.create(**call_params)
            except Exception as e:
                print(f"模型调用失败 - 模型: {model_name}, 错误: {str(e)}")
                print(f"调用参数: {call_params}")
                # 回退到基础参数重试
                basic_params = {k: v for k, v in call_params.items() 
                               if k in ['model', 'messages', 'temperature', 'max_tokens']}
                response = await current_client.chat.completions.create(**basic_params)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    async def reflection_node(self, state: AgentState) -> AgentState:
        """反思评估节点 - 增强版ReAct推理
        
        功能:
        - 评估当前答案质量
        - 进行ReAct综合反思
        - 分析推理链完整性
        - 判断是否需要更多信息
        - 动态调整推理策略
        - 设置反思结果
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态
        """
        print(f"\n=== ReAct增强反思评估节点 ===")
        
        # 如果当前没有答案，直接标记为insufficient
        if not state.get('current_answer'):
            state['reflection_result'] = "insufficient"
            state['messages'].append("🤔 反思评估: 当前没有答案，需要更多信息")
            print("当前没有答案，标记为insufficient")
            return state
        
        # 使用ReAct推理引擎进行综合反思评估
        try:
            comprehensive_reflection = await self.react_reasoning_engine.comprehensive_reflection(state)
            
            # 更新反思结果
            state['reflection_result'] = comprehensive_reflection['reflection_result']
            
            # 记录详细的反思信息
            state['messages'].append(f"🧠 ReAct综合反思: {comprehensive_reflection.get('next_action', {}).get('reason', '已完成反思')}")
            state['messages'].append(f"📊 推理链完整性: {comprehensive_reflection.get('chain_analysis', {}).get('completeness', 'N/A')}")
            state['messages'].append(f"🎯 答案质量评分: {comprehensive_reflection.get('answer_quality', {}).get('score', 'N/A')}")
            state['messages'].append(f"✅ 最终评估: {comprehensive_reflection['reflection_result']}")
            
            # 如果需要调整推理策略
            if comprehensive_reflection.get('next_action', {}).get('action') == 'continue_reasoning':
                # 基于反思结果调整推理策略
                state['messages'].append(f"🔄 推理策略调整: 继续推理过程")
            
            print(f"✅ ReAct综合反思完成:")
            print(f"   评估结果: {comprehensive_reflection['reflection_result']}")
            print(f"   推理质量: {comprehensive_reflection.get('chain_analysis', {}).get('completeness', 'N/A')}")
            print(f"   答案评分: {comprehensive_reflection.get('answer_quality', {}).get('score', 'N/A')}")
            
            return state
            
        except Exception as react_error:
            print(f"⚠️ ReAct综合反思失败，回退到传统评估: {react_error}")
            state['messages'].append(f"⚠️ ReAct反思失败，使用传统评估方式")
            # 继续执行传统反思逻辑
        
        # 准备格式化变量
        used_tools_str = ', '.join(state['used_tools']) if state['used_tools'] else '无'
        
        # 构建反思提示词 - 避免f-string中的中文字符
        user_query = state['query']
        current_answer = state['current_answer']
        
        prompt = f"""你是一个高度智能的答案质量评估员。你的唯一任务是评判一个当前答案是否足够好地回答了用户问题。

用户问题: {user_query}

当前答案: {current_answer}

已使用的检索工具: {used_tools_str}

请从以下几个方面评估答案质量【评估标准】:
1. 完整性: 是否完全回答了用户的问题？
2. 准确性: 答案是否基于可靠的信息？
3. 相关性: 答案是否紧密围绕问题核心？是否包含了不相关的冗余信息？
4. 明确性：答案是否清晰、直接？用户是否能毫不费力地理解？如果答案只是说已找到相关信息但没有给出具体内容，则视为不明确。

指令:
1. 仔细阅读【用户问题】和【当前答案】。
2. 基于上述【评估标准】进行判断。
3. 请严格按照以下JSON格式回复，不要包含任何其他内容:
{{
    "evaluation": "sufficient或insufficient"
}}
    sufficient: 如果答案质量足够高，可以直接呈现给用户。
    insufficient: 如果答案有缺陷（不完整、不明确、部分相关），需要进一步处理或使用其他工具补充信息。"""
        
        try:
            model_name = state.get('model_name') or config.system_config['default_model']
            model_config = config.get_model_config(model_name)
            if not model_config:
                state['reflection_result'] = "insufficient"
                state['messages'].append(f"模型配置错误: {model_name}，无法进行反思评估")
                return state
            
            # 为当前模型获取或创建客户端（使用缓存）
            current_client = self._get_or_create_client(model_name)
            
            # 构建基础参数
            base_params = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": "你是一个专业的答案质量评估专家，能够客观评估答案的完整性和准确性。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            # 使用统一的参数获取方法
            call_params = config.get_model_call_params(model_name, base_params)
            
            try:
                response = await current_client.chat.completions.create(**call_params)
            except Exception as e:
                print(f"模型调用失败 - 模型: {model_name}, 错误: {str(e)}")
                print(f"调用参数: {call_params}")
                # 回退到基础参数重试
                basic_params = {k: v for k, v in call_params.items() 
                               if k in ['model', 'messages', 'temperature', 'max_tokens']}
                response = await current_client.chat.completions.create(**basic_params)
            
            reflection_result = response.choices[0].message.content.strip()
            print(f"反思评估结果: {reflection_result}")
            
            # 简化JSON解析逻辑
            try:
                reflection_data = json.loads(reflection_result)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试提取JSON部分
                if '{' in reflection_result and '}' in reflection_result:
                    start_idx = reflection_result.find('{')
                    end_idx = reflection_result.rfind('}') + 1
                    json_content = reflection_result[start_idx:end_idx]
                    try:
                        reflection_data = json.loads(json_content)
                        print(f"成功提取JSON: {json_content}")
                    except json.JSONDecodeError:
                        print(f"JSON解析失败，使用默认评估")
                        reflection_data = {'evaluation': 'insufficient'}
                else:
                    print(f"未找到JSON格式，使用默认评估")
                    reflection_data = {'evaluation': 'insufficient'}
            
            evaluation = reflection_data.get('evaluation', 'insufficient')
            
            update_result = safe_update_state(state, {'reflection_result': evaluation})
            if not update_result['reflection_result']:
                print("反思节点状态更新警告")
                log_state_issues(state)
            
            state['messages'].append(f"反思评估: {evaluation}")
            
            print(f"评估结果: {evaluation}")
                
        except Exception as e:
            state['reflection_result'] = "insufficient"
            error_msg = f"反思评估失败: {str(e)}"
            state['messages'].append(error_msg)
            print(f"反思评估异常: {e}")
        
        return state
    
    async def final_answer_node(self, state: AgentState) -> AgentState:
        """最终答案生成节点
        
        功能:
        - 展示每个推理步骤的改写查询和初步答案
        - 直接输出最终答案（不包含验证过程）
        - 保存对话记录
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态
        """
        print(f"\n{'='*50}")
        print(f"🎯 最终答案生成节点")
        print(f"{'='*50}")
        
        # 打印当前状态信息
        print(f"📊 状态信息:")
        print(f"  - 用户查询: {state.get('query', '未记录')[:100]}...")
        print(f"  - 消息历史数量: {len(state.get('messages', []))}")
        print(f"  - 检索信息长度: {len(state.get('retrieved_info', ''))} 字符")
        print(f"  - 当前答案长度: {len(state.get('current_answer', ''))} 字符")
        print(f"  - 推理链状态: {'存在' if state.get('reasoning_chain') else '不存在'}")
        print(f"  - 动作历史数量: {len(state.get('action_history', []))}")
        print(f"  - 观察历史数量: {len(state.get('observation_history', []))}")
        print(f"  - 使用工具: {', '.join(state.get('used_tools', []))}")
        
        # 构建详细的推理过程展示
        print(f"\n🔍 构建推理摘要...")
        reasoning_summary = await self._build_reasoning_summary(state)
        print(f"  - 提取到推理步骤数: {reasoning_summary['total_steps']}")
        print(f"  - 使用的工具: {', '.join(reasoning_summary['used_tools'])}")
        
        # 确保有最终答案
        if not state['current_answer']:
            print(f"⚠️ 当前答案为空，使用默认答案")
            state['current_answer'] = f"抱歉，我无法找到足够的信息来回答您的问题: {state['query']}"
        
        # 如果有多个检索结果，可以进行最终整合
        if state['retrieved_info'] and len(state['used_tools']) > 1:
            print(f"🤖 检测到多个工具结果，进行综合答案生成...")
            final_answer = await self._generate_comprehensive_answer(state)
            state['current_answer'] = final_answer
            print(f"  - 综合答案生成完成，长度: {len(final_answer)} 字符")
        
        # 构建完整的输出格式（直接输出最终答案，不包含验证过程）
        print(f"\n📋 格式化输出...")
        formatted_output = self._format_final_output(reasoning_summary, state['current_answer'])
        print(f"  - 格式化输出完成，总长度: {len(formatted_output)} 字符")
        
        # 更新状态中的答案为格式化后的输出
        state['current_answer'] = formatted_output
        state['messages'].append(f"最终答案: {formatted_output}")
        
        # 输出到控制台
        print(f"\n📄 推理过程详细展示:")
        print("\n" + "="*60)
        print("推理过程详细展示:")
        print("="*60)
        print(formatted_output)
        print("="*60)
        
        # 保存助手消息到数据库
        if state.get('conversation_id'):
            try:
                await db.add_message(state['conversation_id'], formatted_output, "assistant")
                print("答案已保存到数据库")
            except Exception as e:
                print(f"保存答案到数据库失败: {str(e)}")
        
        print(f"最终答案生成完成")
        
        # 清理临时状态，避免状态污染
        print("正在清理临时状态...")
        cleaned_state = cleanup_temporary_state(state)
        
        # 验证清理后的状态
        if validate_cleaned_state(cleaned_state):
            print(f"状态清理完成，保留核心字段: query, current_answer, conversation_id")
        else:
            print("警告: 状态清理验证失败，可能存在清理不完整的问题")
            log_state_issues(cleaned_state)
        
        return cleaned_state
    
    async def _build_reasoning_summary(self, state: AgentState) -> Dict[str, Any]:
        """构建推理过程摘要 - 从step_wise_results中获取每一步的详细信息
        
        参数:
            state: 当前状态
            
        返回:
            推理过程摘要字典
        """
        reasoning_summary = {
            'steps': [],
            'total_steps': 0,
            'used_tools': list(state.get('used_tools', [])),
            'original_query': state['query']
        }
        
        # 从step_wise_results中获取每一步的详细信息
        step_wise_results = state.get('step_wise_results', [])
        
        if not step_wise_results:
            # 如果没有step_wise_results，回退到原来的逻辑
            print("⚠️ 没有找到step_wise_results，使用回退逻辑")
            return await self._build_reasoning_summary_fallback(state)
        
        # 智能提取初步答案内容
        def extract_preliminary_answer(answer: str) -> str:
            """提取并格式化初步答案"""
            if not answer:
                return "未获取到答案"
            
            # 如果是网络搜索结果格式
            if '网络搜索结果：' in answer or '搜索结果' in answer:
                lines = answer.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('根据') and not line.startswith('搜索') and not line.startswith('检索'):
                        # 找到第一个实质性内容行
                        if len(line) > 10:  # 确保不是太短的标题
                            return line[:100] + "..." if len(line) > 100 else line
            
            # 对于普通文本，提取前150个字符作为摘要
            if len(answer) > 150:
                # 尝试在句号处截断
                sentences = answer[:150].split('。')
                if len(sentences) > 1:
                    return sentences[0] + '。'
                else:
                    return answer[:150] + "..."
            else:
                return answer
        
        # 构建前两步的推理信息
        max_steps = min(2, len(step_wise_results))
        
        for i in range(max_steps):
            step_data = step_wise_results[i]
            
            step_info = {
                'step_number': step_data.get('step_number', i + 1),
                'rewritten_query': step_data.get('rewritten_query', state['query']),
                'tool_used': step_data.get('selected_tool', '未知工具'),
                'preliminary_answer': extract_preliminary_answer(step_data.get('preliminary_answer', '')),
                'thought': '',
                'confidence': step_data.get('confidence', 0.8)
            }
            reasoning_summary['steps'].append(step_info)
        
        # 如果只有一步但使用了多个工具，尝试创建第二步
        if len(reasoning_summary['steps']) == 1 and len(state.get('used_tools', [])) > 1:
            used_tools_list = list(state.get('used_tools', []))
            step2 = {
                'step_number': 2,
                'rewritten_query': f"使用{used_tools_list[1]}进一步查询: {state['query']}",
                'tool_used': used_tools_list[1],
                'preliminary_answer': extract_preliminary_answer(step_wise_results[0].get('preliminary_answer', '')),
                'thought': '',
                'confidence': 0.8
            }
            reasoning_summary['steps'].append(step2)
        
        reasoning_summary['total_steps'] = len(reasoning_summary['steps'])
        return reasoning_summary
    
    async def _build_reasoning_summary_fallback(self, state: AgentState) -> Dict[str, Any]:
        """回退的推理摘要构建方法 - 当step_wise_results为空时使用
        
        参数:
            state: 当前状态
            
        返回:
            推理过程摘要字典
        """
        reasoning_summary = {
            'steps': [],
            'total_steps': 0,
            'used_tools': list(state.get('used_tools', [])),
            'original_query': state['query']
        }
        
        # 从消息历史中提取改写查询信息
        messages = state.get('messages', [])
        current_answer = state.get('current_answer', '')
        
        # 提取改写查询信息
        rewritten_queries = []
        for message in messages:
            if '🔄 查询改写:' in message:
                query = message.split('🔄 查询改写:')[1].strip()
                rewritten_queries.append(query)
            elif '改写查询:' in message:
                query = message.split('改写查询:')[1].strip()
                rewritten_queries.append(query)
        
        # 如果没有找到改写查询，使用原始查询
        if not rewritten_queries:
            rewritten_queries = [state['query']]
        
        # 提取工具使用信息
        used_tools_list = list(state.get('used_tools', []))
        
        # 从current_answer中提取实际的初步答案内容
        def extract_preliminary_answer_from_current_answer(current_answer: str, step_num: int) -> str:
            """从当前答案中提取初步答案内容"""
            if not current_answer:
                return "未获取到答案"
            
            # 如果是网络搜索结果格式
            if '网络搜索结果：' in current_answer:
                lines = current_answer.split('\n')
                titles = []
                for line in lines:
                    if line.strip() and ('**' in line or line.startswith(str(step_num) + '.')):
                        # 提取标题
                        title = line.replace('**', '').strip()
                        if title.startswith(str(step_num) + '.'):
                            title = title[2:].strip()  # 移除序号
                        if title and len(title) > 5 and not title.startswith('网络搜索结果'):
                            titles.append(title)
                
                if titles:
                    return titles[0] if step_num == 1 else (titles[min(step_num-1, len(titles)-1)] if len(titles) > 1 else titles[0])
            
            # 如果是普通文本，提取前150个字符作为摘要
            if len(current_answer) > 150:
                # 尝试在句号处截断
                sentences = current_answer[:150].split('。')
                if len(sentences) > 1:
                    return sentences[0] + '。'
                else:
                    return current_answer[:150] + "..."
            else:
                return current_answer
        
        # 构建前两步的推理信息
        max_steps = min(2, max(len(rewritten_queries), len(used_tools_list), 1))
        
        for i in range(max_steps):
            step_num = i + 1
            
            # 获取改写查询
            if i < len(rewritten_queries):
                rewritten_query = rewritten_queries[i]
            else:
                rewritten_query = state['query'] if i == 0 else f"基于第{i}步结果进一步查询: {state['query']}"
            
            # 获取使用的工具
            if i < len(used_tools_list):
                tool_used = used_tools_list[i]
            else:
                tool_used = '未知工具'
            
            # 从current_answer中提取实际的初步答案
            preliminary_answer = extract_preliminary_answer_from_current_answer(current_answer, step_num)
            
            step_info = {
                'step_number': step_num,
                'rewritten_query': rewritten_query,
                'tool_used': tool_used,
                'preliminary_answer': preliminary_answer,
                'thought': '',
                'confidence': 0.8  # 默认置信度
            }
            reasoning_summary['steps'].append(step_info)
        
        # 如果只有一步但使用了多个工具，创建第二步
        if len(reasoning_summary['steps']) == 1 and len(used_tools_list) > 1:
            step2 = {
                'step_number': 2,
                'rewritten_query': f"使用{used_tools_list[1]}进一步查询: {state['query']}",
                'tool_used': used_tools_list[1],
                'preliminary_answer': extract_preliminary_answer_from_current_answer(current_answer, 2),
                'thought': '',
                'confidence': 0.8
            }
            reasoning_summary['steps'].append(step2)
        
        reasoning_summary['total_steps'] = len(reasoning_summary['steps'])
        return reasoning_summary
    
    def _format_final_output(self, reasoning_summary: Dict[str, Any], final_answer: str) -> str:
        """格式化最终输出 - 简化版本
        
        参数:
            reasoning_summary: 推理过程摘要
            final_answer: 最终答案
            
        返回:
            格式化的输出字符串
        """
        output_lines = []
        
        # 只显示前两步的关键信息
        steps_to_show = reasoning_summary['steps'][:2]  # 只取前两步
        
        for i, step in enumerate(steps_to_show, 1):
            output_lines.append(f"## 📍 第{i}步推理")
            output_lines.append(f"**🔄 改写查询:** {step['rewritten_query']}")
            
            # 获取简洁的初步答案
            preliminary_answer = step.get('preliminary_answer', '')
            if preliminary_answer:
                # 直接显示已经处理过的初步答案
                output_lines.append(f"**📋 初步答案:** {preliminary_answer}")
            else:
                output_lines.append(f"**📋 初步答案:** 未获取到答案")
            
            output_lines.append("")
        
        # 添加最终答案
        output_lines.append("## ✅ 最终答案")
        output_lines.append(final_answer)
        
        return "\n".join(output_lines)
    
    async def _generate_comprehensive_answer(self, state: AgentState) -> str:
        """生成综合答案
        
        当使用了多个工具时，整合所有信息生成最终答案
        
        参数:
            state: 当前状态
            
        返回:
            综合答案
        """
        # 构建对话历史信息
        conversation_history = "\n".join(state.get('messages', []))
        
        # 构建所有检索信息的汇总
        all_retrieved_info = state.get('retrieved_info', '')
        used_tools_info = f"已使用的检索工具: {', '.join(state.get('used_tools', []))}" if state.get('used_tools') else "未使用检索工具"
        
        prompt = f"""你是一位资深的企业信息专家，你的任务是根据提供的【完整对话历史】和【所有检索信息】以及【当前生成的答案】，为【用户问题】生成一个全面、清晰、结构化的最终答案。

用户问题: {state['query']}

对话历史:
{conversation_history}

{used_tools_info}

所有检索到的信息:
{all_retrieved_info}

当前生成的答案: {state.get('current_answer', '暂无')}

请基于以上所有信息，生成一个准确、完整、结构清晰的最终答案。要求:
1. 充分整合对话历史中的所有相关信息
2. 利用所有检索到的信息源
3. 保持答案的逻辑性和连贯性
4. 如果不同信息源有冲突，请明确说明并给出合理解释
5. 突出最重要和最相关的信息
6. 确保答案直接回答用户的核心问题"""
        
        try:
            model_name = state.get('model_name') or config.system_config['default_model']
            model_config = config.get_model_config(model_name)
            if not model_config:
                return state['current_answer']
            
            # 为当前模型获取或创建客户端（使用缓存）
            current_client = self._get_or_create_client(model_name)
            
            # 构建基础参数
            base_params = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": "你是一个智能助手，擅长整合多个信息源，生成准确、完整的综合答案。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            # 使用统一的参数获取方法
            call_params = config.get_model_call_params(model_name, base_params)
            
            try:
                response = await current_client.chat.completions.create(**call_params)
            except Exception as e:
                print(f"模型调用失败 - 模型: {model_name}, 错误: {str(e)}")
                print(f"调用参数: {call_params}")
                # 回退到基础参数重试
                basic_params = {k: v for k, v in call_params.items() 
                               if k in ['model', 'messages', 'temperature', 'max_tokens']}
                response = await current_client.chat.completions.create(**basic_params)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"生成综合答案失败: {str(e)}")
            return state['current_answer']
