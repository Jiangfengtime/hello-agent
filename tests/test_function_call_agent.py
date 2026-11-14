import json
import secrets
from typing import List, Dict, Any

from dotenv import load_dotenv

from agents.function_call_agent import FunctionCallAgent
from core.llm import HelloAgentLLM
from tools.base import Tool, ToolParameter
from tools.builtin.calcultor import CalculatorTool
from tools.registry import ToolRegistry


class PriceLookup(Tool):
    """本地真实价格查询: 价格运行时随机生成, 模型必须调用工具才能知道"""
    def __init__(self):
        super().__init__("price_lookup", '查询商品SKU的实际单价, 必须使用本工具获得价格')
        self.price = 31 * secrets.randbelow(59)
        self.executions = []

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="sku", type="string", description="商品编号, 如demo-001")]

    def run(self, parameters: Dict[str, Any]) -> str:
        self.executions.append(dict(parameters=parameters))
        if parameters.get('sku') != 'demo-001':
            return json.dumps({"error": "商品不存在"}, ensure_ascii=False)
        result = json.dumps({"sku": "demo-001", "price": self.price}, ensure_ascii=False)
        print("\n[实际价格查询] " + result)
        return result

def test_mult_tool_agent():
    load_dotenv(verbose=True)
    price_tool = PriceLookup()
    calculator = CalculatorTool()
    registry = ToolRegistry()
    registry.register_tool(calculator)
    registry.register_tool(price_tool)
    agent = FunctionCallAgent(
        "原生工具助手",
        llm=HelloAgentLLM(),
        tool_registry=registry,
        max_tool_iterations=4,
        system_prompt="必须先调用price_lookup查询商品实际单价, 再调用python_calculator计算总价, 最后回答. 不要猜价格."
    )

    result = agent.run("查询demo-001的单价, 买4件的总价是多少? 必须实际调用两个工具")
    print(result)