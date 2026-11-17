from dotenv import load_dotenv

from agents.react_agent import ReactAgent
from core.llm import HelloAgentLLM
from tools.builtin.calcultor import CalculatorTool
from tools.registry import ToolRegistry


def test_react_agent():
    load_dotenv(verbose=True)
    calculator = CalculatorTool()
    registry = ToolRegistry()
    registry.register_tool(calculator)
    agent = ReactAgent("原生工具助手", llm=HelloAgentLLM(), tool_registry=registry, max_steps=4)

    result = agent.run("必须先调用python_calculator计算137*29，取得Observation后再用Finish返回最终数值。")
    print(result)