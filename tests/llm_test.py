
from dotenv import load_dotenv

from core.llm import HelloAgentLLM

load_dotenv()

def test_invoke():
    llm = HelloAgentLLM()
    result = llm.invoke([{"role": "user", "content": "请简短问候"}])
    print(result)

def test_think():
    llm = HelloAgentLLM()

    chunks = []
    for chunk in llm.think([
        {"role": "user", "content": "详细讲解一下LLM"}
    ]):
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    print()
    assert "".join(chunks).strip()







