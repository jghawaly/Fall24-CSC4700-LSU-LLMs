from openai import OpenAI
from dotenv import load_dotenv
import os


# load environmental variables
load_dotenv('../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# use GPT-40-mini to answer a simple question in a certain persona
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role": "system", "content": "You are a patient and helpful AI assistant named BobGPT. You speak like a pirate."},
        {"role": "user", "content": "How much wood could a woodchuck chuck if a woodchuck could chuck wood?"}
    ],
    temperature=0.7,
    top_p=1,
    max_tokens=250,
    n=1
)

# print the results
print("Model's Response:")
print('\t', response.choices[0].message.content)
print()
print(f"Input Tokens:  {response.usage.prompt_tokens}")
print(f"Output Tokens: {response.usage.completion_tokens}")
print(f"Cost: ${response.usage.prompt_tokens * 0.15 / 1E6 + response.usage.completion_tokens * 0.6 / 1E6}")
