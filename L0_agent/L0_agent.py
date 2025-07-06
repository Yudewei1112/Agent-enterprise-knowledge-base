"""LangGraph Agent主模块

该模块定义了基于LangGraph的反思型检索Agent，整合了：
- 状态管理
- 节点定义
- 路由逻辑
- 图构建
"""

import asyncio
from typing import Dict, Any, AsyncGenerator
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

# 统一使用绝对导入，避免类型检查问题
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from L0_agent_state import AgentState, create_initial_state
from L0_agent_nodes import AgentNodes
from L0_agent_tools import AgentToolManager
from L0_agent_router import AgentRouter, route_after_intent_analysis, route_after_reflection, route_start, route_after_continue_react_reasoning
from react_reasoning_engine import ReActReasoningEngine
from config import config
from database import db


class LangGraphAgent:
    """基于LangGraph的反思型检索Agent
    
    该Agent具备以下能力：
    1. 意图分析和工具选择
    2. 行动记忆（记住使用过的工具）
    3. 反思能力（评估答案质量）
    4. 自动出环（熔断机制）
    5. 多轮迭代优化
    """
    
    def __init__(self):
        """初始化Agent"""
        self.client = None
        self.tool_manager = None
        self.nodes = None
        self.graph = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        # 初始化OpenAI客户端
        default_model_config = config.get_model_config(config.default_model)
        self.client = AsyncOpenAI(
            api_key=default_model_config["api_key"],
            base_url=default_model_config["api_base"]
        )
        
        # 初始化工具管理器
        self.tool_manager = AgentToolManager()
        
        # 初始化路由器
        self.router = AgentRouter()
        
        # 初始化推理引擎
        self.react_reasoning_engine = ReActReasoningEngine()

        # 初始化节点（传入推理引擎）
        self.nodes = AgentNodes(self.tool_manager, self.react_reasoning_engine)
        
        # 构建图
        self._build_graph()
    
    def _build_graph(self):
        """构建LangGraph图结构"""
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("analyze_intent", self.nodes.analyze_intent_node)
        workflow.add_node("execute_tool", self.nodes.tool_executor_node)
        workflow.add_node("reflect_answer", self.nodes.reflection_node)
        workflow.add_node("generate_final_answer", self.nodes.final_answer_node)
        workflow.add_node("continue_react_reasoning", self.nodes.continue_react_reasoning_node)

        # 设置入口点
        workflow.set_entry_point("analyze_intent")

        # 添加条件边
        workflow.add_conditional_edges(
            "analyze_intent",
            route_after_intent_analysis,
            {
                "execute_tool": "execute_tool",
                "generate_final_answer": "generate_final_answer"
            }
        )

        # 从工具执行节点的条件路由
        workflow.add_conditional_edges(
            "execute_tool",
            self.router.route_after_tool_execution,
            {
                "continue_react_reasoning": "continue_react_reasoning",
                "reflect_answer": "reflect_answer",
                "generate_final_answer": "generate_final_answer"
            }
        )

        # 多步推理节点的条件路由
        workflow.add_conditional_edges(
            "continue_react_reasoning",
            route_after_continue_react_reasoning,
            {
                "execute_tool": "execute_tool",
                "reflect_answer": "reflect_answer",
                "generate_final_answer": "generate_final_answer"
            }
        )
        
        # 反思后的条件路由
        workflow.add_conditional_edges(
            "reflect_answer",
            route_after_reflection,
            {
                "analyze_intent": "analyze_intent",
                "generate_final_answer": "generate_final_answer",
                "end_with_best_effort": "generate_final_answer"
            }
        )
        
        # 最终答案节点连接到结束
        workflow.add_edge("generate_final_answer", END)
        
        # 编译图
        self.graph = workflow.compile()
        
        print("LangGraph Agent 初始化完成")
    
    async def query(self, 
                   query: str, 
                   conversation_id: str = None,
                   model_name: str = None,
                   specific_file: str = None) -> AsyncGenerator[str, None]:
        """处理用户查询（流式输出）
        
        参数:
            query: 用户查询
            conversation_id: 对话ID
            model_name: 模型名称
            specific_file: 指定文件
            
        生成:
            流式响应内容
        """
        print(f"\n{'='*50}")
        print(f"开始处理查询: {query}")
        print(f"对话ID: {conversation_id}")
        print(f"模型: {model_name}")
        print(f"指定文件: {specific_file}")
        print(f"{'='*50}")
        
        # 创建初始状态
        initial_state = create_initial_state(
            query=query,
            conversation_id=conversation_id,
            model_name=model_name,
            specific_file=specific_file
        )
        
        # 保存用户消息到数据库
        if conversation_id:
            try:
                await db.add_message(conversation_id, query, "user")
            except Exception as e:
                print(f"保存用户消息失败: {str(e)}")
        
        try:
            # 执行图
            final_state = None
            async for state in self.graph.astream(initial_state):
                # 实时输出处理进度
                for node_name, node_state in state.items():
                    if node_name != "__end__":
                        print(f"\n--- 节点 {node_name} 执行完成 ---")
                        
                        # 流式输出当前进度
                        if node_state.get('current_answer'):
                            progress_msg = f"正在处理中...\n当前进度: {node_name}\n"
                            yield f"data: {{\"content\": \"{self._escape_json_string(progress_msg)}\", \"type\": \"progress\"}}\n\n"
                        
                        final_state = node_state
            
            # 获取最终答案
            if final_state and final_state.get('current_answer'):
                final_answer = final_state['current_answer']
                
                # 添加处理摘要
                summary = self._generate_process_summary(final_state)
                if summary:
                    final_answer += f"\n\n---\n**处理摘要:**\n{summary}"
                
                print(f"\n最终答案: {final_answer}")
                
                # 流式输出最终答案
                yield f"data: {{\"content\": \"{self._escape_json_string(final_answer)}\", \"type\": \"answer\"}}\n\n"
            else:
                error_msg = "抱歉，处理过程中出现问题，无法生成答案。"
                yield f"data: {{\"content\": \"{error_msg}\", \"type\": \"error\"}}\n\n"
            
            # 发送完成信号
            yield f"data: {{\"done\": true}}\n\n"
            
        except Exception as e:
            error_msg = f"Agent执行失败: {str(e)}"
            print(f"Agent执行异常: {e}")
            yield f"data: {{\"content\": \"{error_msg}\", \"type\": \"error\"}}\n\n"
            yield f"data: {{\"done\": true}}\n\n"
    
    async def query_sync(self, 
                        query: str, 
                        conversation_id: str = None,
                        model_name: str = None,
                        specific_file: str = None) -> str:
        """处理用户查询（同步输出）
        
        参数:
            query: 用户查询
            conversation_id: 对话ID
            model_name: 模型名称
            specific_file: 指定文件
            
        返回:
            最终答案
        """
        # 创建初始状态
        initial_state = create_initial_state(
            query=query,
            conversation_id=conversation_id,
            model_name=model_name,
            specific_file=specific_file
        )
        
        # 保存用户消息到数据库
        if conversation_id:
            try:
                await db.add_message(conversation_id, query, "user")
            except Exception as e:
                print(f"保存用户消息失败: {str(e)}")
        
        try:
            # 执行图
            final_state = await self.graph.ainvoke(initial_state)
            
            # 获取最终答案
            if final_state and final_state.get('current_answer'):
                final_answer = final_state['current_answer']
                
                # 添加处理摘要
                summary = self._generate_process_summary(final_state)
                if summary:
                    final_answer += f"\n\n---\n**处理摘要:**\n{summary}"
                
                return final_answer
            else:
                return "抱歉，处理过程中出现问题，无法生成答案。"
                
        except Exception as e:
            print(f"Agent执行异常: {e}")
            return f"Agent执行失败: {str(e)}"
    
    def _generate_process_summary(self, state: AgentState) -> str:
        """生成处理过程摘要
        
        参数:
            state: 最终状态
            
        返回:
            处理摘要
        """
        summary_parts = []
        
        # 使用的工具
        if state['used_tools']:
            tools_str = ', '.join(state['used_tools'])
            summary_parts.append(f"使用工具: {tools_str}")
        
        # 迭代次数
        if state['iteration_count'] > 1:
            summary_parts.append(f"迭代次数: {state['iteration_count']}")
        
        # 反思结果
        if state.get('reflection_result'):
            reflection_map = {
                'sufficient': '答案质量满足要求',
                'insufficient': '答案质量需要改进'
            }
            reflection_result = state['reflection_result']
            reflection_desc = reflection_map.get(reflection_result, reflection_result if reflection_result else '未知')
            summary_parts.append(f"质量评估: {reflection_desc}")
        
        # 重试信息
        if state['tool_retry_counts']:
            retry_info = []
            for tool, count in state['tool_retry_counts'].items():
                if count > 0:
                    retry_info.append(f"{tool}({count}次)")
            if retry_info:
                summary_parts.append(f"重试记录: {', '.join(retry_info)}")
        
        return ' | '.join(summary_parts) if summary_parts else ""
    
    def _escape_json_string(self, text: str) -> str:
        """转义JSON字符串中的特殊字符
        
        参数:
            text: 原始文本
            
        返回:
            转义后的文本
        """
        if not text:
            return ""
        
        # 转义特殊字符
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        return text
    
    def get_graph_visualization(self) -> str:
        """获取图的可视化表示
        
        返回:
            图的文本描述
        """
        return """
LangGraph Agent 流程图:

[开始] → [意图分析] → [工具执行] → [反思评估] → [决策]
                ↓           ↑ ↓                      ↓
            [最终答案] ←─────┘ └─[重试]              [继续迭代]
                ↓                                    ↑
             [结束]                                  │
                                                    │
                                            [新工具选择]

节点说明:
- 意图分析: 分析用户查询，选择合适的工具
- 工具执行: 执行选定的工具，获取信息（失败时直接重试，不重新分析意图）
- 反思评估: 评估答案质量，决定是否需要更多信息
- 最终答案: 生成并返回最终答案

特性:
- 行动记忆: 记住已使用的工具，避免重复
- 反思能力: 自动评估答案质量
- 直接重试: 工具失败时直接重试，提升效率
- 熔断机制: 防止无限循环，最多重试3次
- 多轮迭代: 根据需要使用多个工具增强答案
"""


# 全局Agent实例
agent = LangGraphAgent()


# 便捷函数
async def process_query_stream(query: str, 
                              conversation_id: str = None,
                              model_name: str = None,
                              specific_file: str = None) -> AsyncGenerator[str, None]:
    """处理查询（流式）"""
    async for chunk in agent.query(query, conversation_id, model_name, specific_file):
        yield chunk


async def process_query(query: str, 
                       conversation_id: str = None,
                       model_name: str = None,
                       specific_file: str = None) -> str:
    """处理查询（同步）"""
    return await agent.query_sync(query, conversation_id, model_name, specific_file)