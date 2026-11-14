import json

from dotenv import load_dotenv

from agents.simple_agent import SimpleAgent
from core.llm import HelloAgentLLM
from tools.builtin.calcultor import CalculatorTool
from tools.registry import ToolRegistry


# 简单对话
def test_simple_agent():
    load_dotenv(verbose=True)
    agent = SimpleAgent("计算助手", llm=HelloAgentLLM())

    result = agent.run("计算 3 * 4 + 6 / 3")
    print("执行结果: ", result)

# 多轮对话
def test_multi_turn_agent():
    load_dotenv(verbose=True)
    agent = SimpleAgent("多轮对话助手", llm=HelloAgentLLM(), system_prompt= "请记住用户的信息, 简短回答.")
    response = agent.run("你好, 我叫小明, 给我推荐一下上海的著名景点")
    print("第一轮对话: \n", response)

    response = agent.run("还记得我的名字以为我咨询的问题吗")
    print("第二轮对话: \n", response)

# 流式对话
def test_stream_agent():
    load_dotenv(verbose=True)

    agent = SimpleAgent("流式对话助手", llm=HelloAgentLLM(), system_prompt="简洁回答")

    for chunk in agent.stream_run("请介绍一下SSE协议"):
        print(chunk, end="", flush=True)
    print()

# 工具调用
def test_tool_agent():
    load_dotenv(verbose=True)
    calculator = CalculatorTool()
    registry = ToolRegistry()
    registry.register_tool(calculator)
    agent = SimpleAgent(
        "文本工具助手",
        llm=HelloAgentLLM(),
        tool_registry=registry,
        system_prompt="数学题必须使用python_calculator工具, 收到执行结果再回答"
    )
    answer = agent.run("请使用工具计算 137*29, 取得结果后再输出最终数值。", max_tool_iterations=2)
    print(answer)