"""Agent状态定义模块

该模块定义了LangGraph Agent的状态结构，包括：
- 用户查询和对话历史
- 工具使用记录和重试计数
- 当前答案和反思结果
"""

from typing import TypedDict, List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    content: str
    error: Optional[str] = None
    tool_name: str = ""


@dataclass
class ReActStep:
    """ReAct推理步骤"""
    step_id: int
    thought: str  # 思考过程
    action: str  # 执行的动作
    action_input: Dict[str, Any]  # 动作输入参数
    observation: str  # 观察结果
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0  # 置信度评分
    

@dataclass
class ReasoningChain:
    """推理链管理"""
    chain_id: str
    steps: List[ReActStep] = field(default_factory=list)
    current_step: int = 0
    goal: str = ""
    status: str = "active"  # active, completed, failed
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他推理链
    
    def add_step(self, thought: str, action: str, action_input: Dict[str, Any], observation: str = "") -> ReActStep:
        """添加推理步骤"""
        step = ReActStep(
            step_id=len(self.steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation
        )
        self.steps.append(step)
        self.current_step = len(self.steps)
        return step
    
    def get_current_step(self) -> Optional[ReActStep]:
        """获取当前步骤"""
        if self.steps:
            return self.steps[-1]
        return None
    
    def update_observation(self, observation: str, confidence: float = 0.0):
        """更新最新步骤的观察结果"""
        if self.steps:
            self.steps[-1].observation = observation
            self.steps[-1].confidence = confidence


class AgentState(TypedDict):
    """Agent状态定义
    
    这个状态会在整个图运行过程中传递和更新
    """
    # 原始用户问题
    query: str
    
    # 改写后的查询（包含历史会话信息）
    rewritten_query: str
    
    # 整个对话历史，用于支持多轮对话和传递中间结果
    messages: List[str]
    
    # 记录已经使用过的工具，避免重复低效的检索
    # 格式: ["tool_name_1", "tool_name_2"]
    used_tools: List[str]
    
    # 记录每个工具的失败次数，用于熔断
    # 格式: {"tool_name": count}
    tool_retry_counts: Dict[str, int]
    
    # 最新一次生成的答案，用于反思
    current_answer: str
    
    # 反思后的评估结果 ("sufficient", "insufficient")
    reflection_result: str
    
    # 对话ID，用于保存消息
    conversation_id: Optional[str]
    
    # 模型名称
    model_name: Optional[str]
    
    # 指定文件（用于本地检索）
    specific_file: Optional[str]
    
    # 检索到的信息汇总
    retrieved_info: str
    
    # 循环计数器，防止无限循环
    iteration_count: int
    
    # 意图分析选择的工具（临时状态）
    selected_tool: Optional[str]
    
    # 工具执行参数（临时状态）
    tool_parameters: Optional[Dict[str, Any]]
    
    # 工具执行结果列表
    tool_results: List[ToolResult]
    
    # 工具执行状态（临时状态）
    tool_execution_status: Optional[str]
    
    # === ReAct推理增强字段 ===
    # 思考历史记录
    thought_history: List[str]
    
    # 行动历史记录
    action_history: List[Dict[str, Any]]
    
    # 观察历史记录
    observation_history: List[str]
    
    # 当前ReAct步骤编号
    react_step: int
    
    # 推理链管理
    reasoning_chain: Optional[ReasoningChain]
    
    # 步骤依赖关系
    step_dependencies: Dict[str, List[str]]
    
    # 推理规划
    reasoning_plan: List[Dict[str, Any]]
    
    # 当前推理目标
    current_reasoning_goal: str
    
    # 推理置信度
    reasoning_confidence: float
    
    # 多步骤推理标记
    multi_step_reasoning: bool
    
    # 当前步骤索引（多步骤推理用）
    current_step_index: int
    
    # 最后一次工具执行结果（用于多步骤推理）- 修正类型为ToolResult
    last_tool_result: Optional[ToolResult]
    
    # ReAct步骤是否为最终步骤（新增字段）
    react_step_is_final: bool
    
    # 分步推理记录 - 解决变量覆盖问题
    step_wise_results: List[Dict[str, Any]]
    
    # 复杂度判断字段
    query_complexity: str  # 查询复杂度: "simple", "medium", "complex"
    reasoning_strategy: str  # 推理策略: "direct", "simplified", "multi_hop"
    complexity_confidence: float  # 复杂度评估置信度
    complexity_factors: List[str]  # 复杂度关键因素
    
    # === 简单工作流字段 ===
    simple_workflow_active: bool  # 简单工作流是否激活
    simple_workflow_step: int  # 当前简单工作流步骤
    simple_workflow_retry_count: int  # 简单工作流重试计数
    simple_tool_retry_count: int  # 简单工作流工具重试计数
    simple_workflow_mode: bool  # 标记使用简单工作流模式
    



def create_initial_state(query: str, conversation_id: Optional[str] = None, 
                        model_name: Optional[str] = None,
                        specific_file: Optional[str] = None) -> AgentState:
    """创建初始状态
    
    参数:
        query: 用户查询
        conversation_id: 对话ID
        model_name: 模型名称
        specific_file: 指定文件
        
    返回:
        初始化的Agent状态
    """
    # 创建初始推理链
    initial_chain = ReasoningChain(
        chain_id=f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        goal=query,
        status="active"
    )
    
    return AgentState(
        query=query,
        rewritten_query=query,  # 初始时与原始查询相同
        messages=[f"用户问题: {query}"],
        tool_retry_counts={},
        current_answer="",
        reflection_result="",
        conversation_id=conversation_id,
        model_name=model_name,
        specific_file=specific_file,
        retrieved_info="",
        iteration_count=0,
        selected_tool=None,
        tool_parameters=None,
        tool_results=[],
        tool_execution_status=None,
        # ReAct推理字段初始化
        thought_history=[],
        action_history=[],
        observation_history=[],
        react_step=0,
        reasoning_chain=initial_chain,
        step_dependencies={},
        reasoning_plan=[],
        current_reasoning_goal=query,
        reasoning_confidence=0.0,
        multi_step_reasoning=False,
        current_step_index=0,
        last_tool_result=None,
        react_step_is_final=False,  # 新增字段初始化
        step_wise_results=[],  # 初始化分步推理记录
        # 简单工作流字段初始化
        simple_workflow_active=False,
        simple_workflow_step=0,
        simple_workflow_retry_count=0,
        simple_tool_retry_count=0,
        simple_workflow_mode=False,
        used_tools=[]  # 工具记忆初始化
    )


def cleanup_temporary_state(state: AgentState) -> AgentState:
    """清理临时状态
    
    在用户查询完成、生成最终答案后调用，清理临时状态字段，
    避免状态污染，同时保留必要的对话信息。
    
    清理的临时字段包括:
    - selected_tool: 意图分析选择的工具
    - tool_parameters: 工具执行参数
    - tool_results: 工具执行结果列表
    - retrieved_info: 检索到的信息汇总
    - reflection_result: 反思评估结果
    - used_tools: 已使用工具记录
    - tool_retry_counts: 工具重试计数
    - iteration_count: 迭代计数器
    
    保留的核心字段:
    - query: 用户查询
    - current_answer: 最终答案
    - conversation_id: 对话ID
    - model_name: 模型名称
    - specific_file: 指定文件
    - messages: 对话历史（保留关键消息）
    
    参数:
        state: 当前状态
        
    返回:
        清理后的状态
    """
    # 保留关键的对话消息
    essential_messages = []
    for msg in state['messages']:
        # 保留用户问题和最终答案
        if (msg.startswith("用户问题:") or 
            msg.startswith("最终答案:") or
            "最终答案" in msg):
            essential_messages.append(msg)
    
    # 创建清理后的状态
    cleaned_state = AgentState(
        query=state['query'],
        rewritten_query=state['rewritten_query'],  # 保留改写后的查询
        messages=essential_messages,
        used_tools=[],  # 清理
        tool_retry_counts={},  # 清理
        current_answer=state['current_answer'],
        reflection_result="",  # 清理
        conversation_id=state['conversation_id'],
        model_name=state['model_name'],
        specific_file=state['specific_file'],
        retrieved_info="",  # 清理
        iteration_count=0,  # 重置
        selected_tool=None,  # 清理
        tool_parameters=None,  # 清理
        tool_results=[],  # 清理
        tool_execution_status=None,  # 清理
        # ReAct推理字段清理
        thought_history=[],  # 清理思考历史
        action_history=[],  # 清理行动历史
        observation_history=[],  # 清理观察历史
        react_step=0,  # 重置步骤
        reasoning_chain=None,  # 清理推理链
        step_dependencies={},  # 清理依赖关系
        reasoning_plan=[],  # 清理推理规划
        current_reasoning_goal="",  # 清理当前目标
        reasoning_confidence=0.0,  # 重置置信度
        multi_step_reasoning=False,  # 重置多步骤推理标记
        current_step_index=0,  # 重置步骤索引
        last_tool_result=None,  # 清理最后工具结果
        react_step_is_final=False  # 新增字段清理
    )
    
    return cleaned_state





