
class HelloAgentException(Exception):
    """HelloAgent基础异常类"""
    pass

class LLMException(HelloAgentException):
    """LLM相关异常"""
    pass

class AgentException(HelloAgentException):
    """Agent相关异常"""
    pass

class ConfigException(HelloAgentException):
    """配置相关异常"""
    pass

class ToolException(HelloAgentException):
    """工具相关异常"""
    pass
