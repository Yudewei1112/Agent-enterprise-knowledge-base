"""L0_agent包初始化模块

L0_agent是企业知识库系统的核心Agent，具备以下功能：
- 意图分析和工具选择
- 多工具协调执行
- 反思评估和迭代优化
- 熔断机制和错误处理

主要组件：
- L0_agent: 主Agent类
- L0_agent_state: 状态管理
- L0_agent_nodes: 节点定义
- L0_agent_router: 路由逻辑
- L0_agent_tools: 工具封装
- agent_state_validator: 状态验证
"""

# 使用相对导入避免循环导入问题
from .L0_agent import LangGraphAgent
from .L0_agent_state import AgentState, ToolResult, create_initial_state, cleanup_temporary_state
from .L0_agent_nodes import AgentNodes
from .L0_agent_router import AgentRouter, route_after_intent_analysis, route_after_reflection, route_start
from .L0_agent_tools import AgentToolManager
from .agent_state_validator import safe_update_state, validate_state_integrity, log_state_issues, validate_cleaned_state

__version__ = "1.0.0"
__author__ = "Enterprise Knowledge Base Team"

# 导出主要类和函数
__all__ = [
    # 主要类
    "LangGraphAgent",
    "AgentNodes", 
    "AgentRouter",
    "AgentToolManager",
    
    # 状态相关
    "AgentState",
    "ToolResult", 
    "create_initial_state",
    "cleanup_temporary_state",
    
    # 路由函数
    "route_after_intent_analysis",
    "route_after_reflection", 
    "route_start",
    
    # 状态验证
    "safe_update_state",
    "validate_state_integrity",
    "log_state_issues",
    "validate_cleaned_state"
]