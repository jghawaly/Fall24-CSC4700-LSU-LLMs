import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

# load environmental variables
load_dotenv('../.env')


client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a patient and helpful AI assistant named BobGPT. You speak like a pirate."),
        UserMessage(content="How much wood could a woodchuck chuck if a woodchuck could chuck wood?"),
    ]
)

# print the results
print("Model's Response:")
print('\t', response.choices[0].message.content)
print()
print(f"Input Tokens:  {response.usage.prompt_tokens}")
print(f"Output Tokens: {response.usage.completion_tokens}")
print(f"Cost: ${response.usage.prompt_tokens * 0.0003 / 1000 + response.usage.completion_tokens * 0.00061 / 1000}")
