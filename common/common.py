from typing import Literal

SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
    "custom"
]

MessageRole = Literal[
    "user",
    "assistant",
    "system",
    "tool"
]