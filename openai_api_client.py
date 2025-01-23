from openai import OpenAI

client = OpenAI(
    api_key='-',
    base_url = 'http://0.0.0.0:6006'
)

messages = [
    {'role': 'user', 'content': "Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.</s>"}
]
response = client.chat.completions.create(
    model='model',
    messages=messages,
    temperature=0,
    max_tokens=1024,
)
print(response)