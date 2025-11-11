from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

resp = client.chat.completions.create(
    model="qwen/qwen-vl-plus",
    messages=[{"role": "user", "content": "Hello"}]
)

print(resp.choices[0].message.content)
