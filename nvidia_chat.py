import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

completion = client.chat.completions.create(
    model="qwen/qwen2.5-coder-32b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=False
)

print(completion.choices[0].message)
