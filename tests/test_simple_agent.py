from dotenv import load_dotenv

from agents.simple_agent import SimpleAgent
from core.llm import HelloAgentLLM


def test_simple_agent():
    load_dotenv(verbose=True)
    agent = SimpleAgent("计算助手", llm=HelloAgentLLM())

    result = agent.run("计算 3 * 4 + 6 / 3")
    print("执行结果: ", result)