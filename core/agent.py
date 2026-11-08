from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from core.config import Config
from core.llm import HelloAgentLLM
from core.message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        # 对象级对话历史, 只存在于当前Agent实例内, 不等于外部长期记忆.
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史消息"""
        self._history.append(message)

    def clear_history(self):
        """清空历史消息"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史消息"""
        # 仅复制列表容器; 其中的Message对象仍是原对象
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()
