"""Agent状态验证器模块

该模块提供类型安全的状态操作工具，确保所有状态字段都符合预期类型。"""

from typing import Any, Dict, List, Set, Optional, Union, get_type_hints
import logging
# 统一使用绝对导入，避免类型检查问题
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from L0_agent_state import AgentState

# 定义所有合法的状态字段及其类型
VALID_STATE_FIELDS = {
    'query': str,
    'messages': list,
    'used_tools': set,
    'tool_retry_counts': dict,
    'current_answer': str,
    'reflection_result': str,
    'conversation_id': (str, type(None)),
    'model_name': (str, type(None)),
    'specific_file': (str, type(None)),
    'retrieved_info': str,
    'iteration_count': int,
    'selected_tool': (str, type(None)),
    'tool_parameters': (dict, type(None)),
    'tool_results': list
}


class StateValidator:
    """状态验证器
    
    提供类型安全的状态操作方法，确保状态字段的类型正确性。
    """
    
    def __init__(self):
        """初始化状态验证器"""
        self.logger = logging.getLogger(__name__)
        # 获取 AgentState 的类型提示
        self._type_hints = get_type_hints(AgentState)
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """验证状态字典是否符合 AgentState 类型定义
        
        参数:
            state: 要验证的状态字典
            
        返回:
            验证是否通过
        """
        try:
            for field_name, expected_type in self._type_hints.items():
                if field_name in state:
                    value = state[field_name]
                    if not self._check_type(value, expected_type, field_name):
                        return False
            return True
        except Exception as e:
            self.logger.error(f"状态验证异常: {e}")
            return False
    
    def _check_type(self, value: Any, expected_type: type, field_name: str) -> bool:
        """检查值是否符合预期类型
        
        参数:
            value: 要检查的值
            expected_type: 预期类型
            field_name: 字段名称
            
        返回:
            类型检查是否通过
        """
        try:
            # 处理 Optional 类型
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                # 检查是否是 Optional[T] (即 Union[T, None])
                args = expected_type.__args__
                if len(args) == 2 and type(None) in args:
                    if value is None:
                        return True
                    # 获取非 None 的类型
                    non_none_type = args[0] if args[1] is type(None) else args[1]
                    return self._check_basic_type(value, non_none_type, field_name)
            
            return self._check_basic_type(value, expected_type, field_name)
            
        except Exception as e:
            self.logger.warning(f"类型检查异常 {field_name}: {e}")
            return True  # 异常时不阻止执行
    
    def _check_basic_type(self, value: Any, expected_type: type, field_name: str) -> bool:
        """检查基本类型
        
        参数:
            value: 要检查的值
            expected_type: 预期类型
            field_name: 字段名称
            
        返回:
            类型检查是否通过
        """
        # 处理泛型类型
        if hasattr(expected_type, '__origin__'):
            origin = expected_type.__origin__
            
            if origin is list:
                if not isinstance(value, list):
                    self.logger.warning(f"字段 {field_name} 期望 list 类型，实际为 {type(value)}")
                    return False
                return True
            
            elif origin is dict:
                if not isinstance(value, dict):
                    self.logger.warning(f"字段 {field_name} 期望 dict 类型，实际为 {type(value)}")
                    return False
                return True
            
            elif origin is set:
                if not isinstance(value, set):
                    self.logger.warning(f"字段 {field_name} 期望 set 类型，实际为 {type(value)}")
                    return False
                return True
        
        # 基本类型检查
        if not isinstance(value, expected_type):
            self.logger.warning(f"字段 {field_name} 期望 {expected_type} 类型，实际为 {type(value)}")
            return False
        
        return True
    
    def safe_set_field(self, state: Dict[str, Any], field_name: str, value: Any) -> bool:
        """安全设置状态字段
        
        参数:
            state: 状态字典
            field_name: 字段名称
            value: 字段值
            
        返回:
            设置是否成功
        """
        if field_name not in self._type_hints:
            self.logger.warning(f"字段 {field_name} 不在 AgentState 类型定义中")
            return False
        
        expected_type = self._type_hints[field_name]
        if self._check_type(value, expected_type, field_name):
            state[field_name] = value
            return True
        else:
            self.logger.error(f"字段 {field_name} 类型检查失败，拒绝设置")
            return False
    
    def get_missing_fields(self, state: Dict[str, Any]) -> List[str]:
        """获取缺失的必需字段
        
        参数:
            state: 状态字典
            
        返回:
            缺失字段列表
        """
        missing_fields = []
        for field_name, expected_type in self._type_hints.items():
            # 检查是否是 Optional 类型
            is_optional = (hasattr(expected_type, '__origin__') and 
                          expected_type.__origin__ is Union and 
                          type(None) in expected_type.__args__)
            
            if not is_optional and field_name not in state:
                missing_fields.append(field_name)
        
        return missing_fields
    
    def get_undefined_fields(self, state: Dict[str, Any]) -> List[str]:
        """获取未在 AgentState 中定义的字段
        
        参数:
            state: 状态字典
            
        返回:
            未定义字段列表
        """
        undefined_fields = []
        for field_name in state.keys():
            if field_name not in self._type_hints:
                undefined_fields.append(field_name)
        
        return undefined_fields
    
    def generate_state_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成状态报告
        
        参数:
            state: 状态字典
            
        返回:
            状态报告字典
        """
        return {
            "valid": self.validate_state(state),
            "missing_fields": self.get_missing_fields(state),
            "undefined_fields": self.get_undefined_fields(state),
            "total_fields": len(state),
            "defined_fields": len(self._type_hints),
            "field_types": {k: str(type(v)) for k, v in state.items()}
        }


# 全局状态验证器实例
state_validator = StateValidator()


def safe_update_state(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, bool]:
    """安全更新状态字段
    
    参数:
        state: 状态字典
        updates: 要更新的字段字典
        
    返回:
        每个字段的更新结果
    """
    results = {}
    for field_name, value in updates.items():
        results[field_name] = state_validator.safe_set_field(state, field_name, value)
    return results


def validate_state_integrity(state: Dict[str, Any]) -> bool:
    """验证状态完整性
    
    参数:
        state: 状态字典
        
    返回:
        状态是否完整有效
    """
    return state_validator.validate_state(state)


def log_state_issues(state: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """记录状态问题
    
    参数:
        state: 状态字典
        logger: 日志记录器，如果为None则使用默认记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    report = state_validator.generate_state_report(state)
    
    if not report["valid"]:
        logger.warning("状态验证失败")
    
    if report["missing_fields"]:
        logger.warning(f"缺失必需字段: {report['missing_fields']}")
    
    if report["undefined_fields"]:
        logger.warning(f"未定义字段: {report['undefined_fields']}")


def validate_cleaned_state(state: Dict[str, Any]) -> bool:
    """验证清理后的状态
    
    检查状态清理后是否仍然保持有效性，确保核心字段完整。
    
    参数:
        state: 清理后的状态字典
        
    返回:
        状态是否有效
    """
    # 核心必需字段
    required_core_fields = {
        'query': str,
        'current_answer': str,
        'messages': list,
        'conversation_id': (str, type(None)),
        'model_name': (str, type(None)),
        'specific_file': (str, type(None))
    }
    
    # 检查核心字段是否存在且类型正确
    for field, expected_type in required_core_fields.items():
        if field not in state:
            print(f"清理后状态缺失核心字段: {field}")
            return False
        
        if not isinstance(state[field], expected_type):
            print(f"清理后状态字段类型错误: {field}, 期望 {expected_type}, 实际 {type(state[field])}")
            return False
    
    # 检查临时字段是否已被正确清理
    temporary_fields = {
        'selected_tool', 'tool_parameters', 'tool_results', 
        'retrieved_info', 'reflection_result', 'used_tools', 
        'tool_retry_counts'
    }
    
    for field in temporary_fields:
        if field in state:
            value = state[field]
            # 检查是否已被清理为默认值
            if field == 'selected_tool' and value is not None:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
            elif field == 'tool_parameters' and value is not None:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
            elif field == 'tool_results' and value:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
            elif field == 'retrieved_info' and value:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
            elif field == 'reflection_result' and value:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
            elif field == 'used_tools' and value:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
            elif field == 'tool_retry_counts' and value:
                print(f"临时字段未正确清理: {field} = {value}")
                return False
    
    print("状态清理验证通过")
    return True