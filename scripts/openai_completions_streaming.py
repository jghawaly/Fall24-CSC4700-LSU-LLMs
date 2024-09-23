from openai import OpenAI
from dotenv import load_dotenv
import os


# load environmental variables
load_dotenv('../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# use GPT-40-mini to answer a simple question in a certain persona
stream = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role": "system", "content": "You are a patient and helpful AI assistant named BobGPT."},
        {"role": "user", "content": "Give me a list of every one of the Roman emperors, starting with the republic."}
    ],
    temperature=0.7,
    top_p=1,
    max_tokens=5000,
    n=1,
    stream=True
)

print("Model's Response:")
for chunk in stream:
    chunk_content = chunk.choices[0].delta.content
    if chunk_content is not None:
        print(chunk_content, end="")

