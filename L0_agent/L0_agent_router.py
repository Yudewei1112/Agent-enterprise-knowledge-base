"""Agent路由模块

该模块定义了LangGraph中的路由逻辑，包括：
- 条件边函数
- 熔断机制
- 流程控制逻辑
"""

from typing import Literal
# 统一使用绝对导入，避免类型检查问题
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from L0_agent_state import AgentState


class AgentRouter:
    """Agent路由器
    
    负责决定图中节点之间的跳转逻辑
    """
    
    def __init__(self, max_iterations: int = 5, max_tool_retries: int = 3):
        """初始化路由器
        
        参数:
            max_iterations: 最大迭代次数
            max_tool_retries: 单个工具最大重试次数
        """
        self.max_iterations = max_iterations
        self.max_tool_retries = max_tool_retries
    
    def should_continue_after_intent_analysis(self, state: AgentState) -> Literal["execute_tool", "generate_final_answer"]:
        """意图分析后的路由决策
        
        参数:
            state: 当前状态
            
        返回:
            下一个节点名称
        """
        # 检查是否选择了工具
        if state.get('selected_tool'):
            return "execute_tool"
        else:
            # 没有选择工具，直接生成最终答案
            return "generate_final_answer"
    
    def route_after_tool_execution(self, state: AgentState) -> Literal["continue_react_reasoning", "reflect_answer", "generate_final_answer"]:
        """工具执行后的路由决策

        根据ReAct推理步骤是否完成来决定下一步:
        - 如果推理未结束，进入`continue_react_reasoning`节点准备下一步
        - 如果推理已结束，进入`reflect_answer`节点进行反思
        - 如果出现严重错误，直接生成最终答案

        参数:
            state: 当前状态

        返回:
            下一个节点名称
        """
        print(f"\n=== 工具执行后路由决策 ===")
        is_final_step = state.get('react_step_is_final', True)
        print(f"当前步骤是否为最后一步: {is_final_step}")

        if not is_final_step:
            decision = "continue_react_reasoning"
            print(f"🧠 ReAct推理未结束，准备下一步。路由到: {decision}")
        else:
            decision = "reflect_answer"
            print(f"✅ ReAct推理结束，进入反思。路由到: {decision}")
        
        print(f"路由决策: {decision}")
        return decision
    
    def should_continue_after_reflection(self, state: AgentState) -> Literal["analyze_intent", "generate_final_answer", "end_with_best_effort"]:
        """反思评估后的路由决策
        
        这是核心的路由逻辑，决定是否继续迭代、结束流程或强制退出
        
        参数:
            state: 当前状态
            
        返回:
            下一个节点名称或结束标志
        """
        print(f"\n=== 路由决策 ===")
        print(f"反思结果: {state.get('reflection_result')}")
        print(f"已使用工具: {state['used_tools']}")
        print(f"迭代次数: {state['iteration_count']}")
        print(f"工具重试计数: {state['tool_retry_counts']}")
        
        # 增加迭代计数
        state['iteration_count'] += 1
        
        # 检查熔断条件
        circuit_breaker_result = self._check_circuit_breaker(state)
        if circuit_breaker_result:
            print(f"触发熔断机制: {circuit_breaker_result}")
            return "end_with_best_effort"
        
        # 检查反思结果
        reflection_result = state.get('reflection_result', 'insufficient')
        
        if reflection_result == "sufficient":
            print("答案质量满足要求，生成最终答案")
            return "generate_final_answer"
        
        # 答案质量不足，检查是否还有可用工具
        available_tools = self._get_available_tools(state)
        
        if not available_tools:
            print("没有更多可用工具，生成最终答案")
            return "generate_final_answer"
        
        # 检查迭代次数限制
        if state['iteration_count'] >= self.max_iterations:
            print(f"达到最大迭代次数 {self.max_iterations}，强制结束")
            return "end_with_best_effort"
        
        print(f"继续分析，可用工具: {list(available_tools)}")
        return "analyze_intent"
    
    def _check_circuit_breaker(self, state: AgentState) -> str:
        """检查熔断条件
        
        参数:
            state: 当前状态
            
        返回:
            熔断原因，如果没有触发熔断则返回空字符串
        """
        # 检查工具重试次数
        for tool_name, retry_count in state['tool_retry_counts'].items():
            if retry_count >= self.max_tool_retries:
                return f"工具 {tool_name} 失败次数达到 {self.max_tool_retries} 次"
        
        # 检查总迭代次数
        if state['iteration_count'] >= self.max_iterations:
            return f"总迭代次数达到 {self.max_iterations} 次"
        
        # 检查是否存在无限循环的风险
        if len(state['messages']) > 50:  # 消息过多可能表示陷入循环
            return "消息数量过多，可能存在无限循环"
        
        return ""
    
    def _get_available_tools(self, state: AgentState) -> set:
        """获取可用工具列表
        
        参数:
            state: 当前状态
            
        返回:
            可用工具集合
        """
        all_tools = {"local_document_rag_search", "internet_search", "mcp_service_lookup"}
        used_tools = state['used_tools']
        failed_tools = set()
        
        # 排除失败次数过多的工具
        for tool_name, retry_count in state['tool_retry_counts'].items():
            if retry_count >= self.max_tool_retries:
                failed_tools.add(tool_name)
        
        available_tools = all_tools - used_tools - failed_tools
        return available_tools
    
    def route_start(self, state: AgentState) -> Literal["analyze_intent"]:
        """开始节点的路由
        
        参数:
            state: 当前状态
            
        返回:
            下一个节点名称
        """
        print(f"\n=== 开始Agent流程 ===")
        print(f"用户查询: {state['query']}")
        return "analyze_intent"
    
    def should_end_process(self, state: AgentState) -> bool:
        """判断是否应该结束流程
        
        参数:
            state: 当前状态
            
        返回:
            是否结束流程
        """
        # 检查是否有致命错误
        if any("致命错误" in msg for msg in state['messages']):
            return True
        
        # 检查是否已经生成了最终答案
        if state.get('current_answer') and state.get('reflection_result') == "sufficient":
            return True
        
        return False
    
    def get_routing_info(self, state: AgentState) -> dict:
        """获取路由信息，用于调试
        
        参数:
            state: 当前状态
            
        返回:
            路由信息字典
        """
        available_tools = self._get_available_tools(state)
        circuit_breaker_reason = self._check_circuit_breaker(state)
        
        return {
            "iteration_count": state['iteration_count'],
            "used_tools": list(state['used_tools']),
            "available_tools": list(available_tools),
            "tool_retry_counts": dict(state['tool_retry_counts']),
            "reflection_result": state.get('reflection_result'),
            "circuit_breaker_triggered": bool(circuit_breaker_reason),
            "circuit_breaker_reason": circuit_breaker_reason,
            "has_current_answer": bool(state.get('current_answer')),
            "message_count": len(state['messages'])
        }


# 全局路由器实例
router = AgentRouter()


# 路由函数（供LangGraph使用）
def route_after_intent_analysis(state: AgentState) -> Literal["execute_tool", "generate_final_answer"]:
    """意图分析后的路由函数
    
    根据意图分析的结果决定下一步:
    - 如果选择了工具，路由到工具执行节点
    - 否则路由到最终答案生成节点
    
    参数:
        state: 当前状态
        
    返回:
        下一个节点名称
    """
    print(f"\n=== 意图分析后路由 ===")
    print(f"路由函数 - 当前状态键: {list(state.keys())}")
    print(f"路由函数 - 是否有selected_tool: {'selected_tool' in state}")
    
    selected_tool = state.get('selected_tool')
    print(f"路由函数 - selected_tool值: {selected_tool}")
    print(f"路由函数 - selected_tool类型: {type(selected_tool)}")
    
    result = router.should_continue_after_intent_analysis(state)
    print(f"路由决策结果: {result}")
    return result


def route_after_reflection(state: AgentState) -> Literal["analyze_intent", "generate_final_answer", "end_with_best_effort"]:
    """反思评估后的路由函数"""
    return router.should_continue_after_reflection(state)


def route_start(state: AgentState) -> Literal["analyze_intent"]:
    """开始路由函数"""
    return router.route_start(state)


def route_after_continue_react_reasoning(state: AgentState) -> Literal["execute_tool", "reflect_answer", "generate_final_answer"]:
    """continue_react_reasoning节点后的路由函数
    
    根据多步推理的状态决定下一步:
    - 如果还有步骤需要执行，路由到execute_tool
    - 如果推理已完成，路由到reflect_answer
    - 如果出现错误，路由到generate_final_answer
    
    参数:
        state: 当前状态
        
    返回:
        下一个节点名称
    """
    print(f"\n=== continue_react_reasoning后路由决策 ===")
    
    # 检查是否是多步推理
    if not state.get('multi_step_reasoning', False):
        print("非多步推理，路由到反思节点")
        return "reflect_answer"
    
    # 检查是否有选择的工具
    selected_tool = state.get('selected_tool')
    if selected_tool and selected_tool.strip():
        print(f"有选择的工具 {selected_tool}，路由到工具执行节点")
        return "execute_tool"
    
    # 检查推理计划状态
    reasoning_plan = state.get('reasoning_plan', [])
    current_step_index = state.get('current_step_index', 0)
    
    if not reasoning_plan or current_step_index >= len(reasoning_plan):
        print("推理计划已完成，路由到反思节点")
        return "reflect_answer"
    
    # 检查是否有工具执行状态错误
    tool_execution_status = state.get('tool_execution_status')
    if tool_execution_status in ['no_tool_selected', 'tool_not_found', 'max_retries_reached']:
        print(f"工具执行状态异常: {tool_execution_status}，路由到最终答案生成")
        return "generate_final_answer"
    
    print("默认路由到反思节点")
    return "reflect_answer"