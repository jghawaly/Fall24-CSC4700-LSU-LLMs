from openai import OpenAI
from dotenv import load_dotenv
import os
import json


# load environmental variables
load_dotenv('../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# use GPT-40-mini to answer a simple question in a certain persona
response = client.chat.completions.create(
    model='gpt-4o-2024-08-06',
    messages=[
        {"role": "system", "content": "You are an expert in programming Python."},
        {"role": "user", "content": "How do I create a new environment and build a simple neural network in Tensorflow?"}
    ],
    temperature=0.2,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "coding_output",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "Each step of the process",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string",
                                                "description": "An explanation of what this step does and why you chose it."},
                                "output": {"type": "string",
                                           "description": "The command/code and what to do with it"}
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": False
                        }
                    },
                    "final_response": {"type": "string",
                                       "description": "The final response after the set of steps/explanations. This can be used to explain other things the user could do/try."}
                },
                "required": ["steps", "final_response"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
json_object = json.loads(response.choices[0].message.content)

# print the results
print("Model's Response:")
print('\t', json_object)
print("Formatted Response:")
for step_obj in json_object['steps']:
    print(step_obj['output'])
    print(f"\t Explanation: {step_obj['explanation']}")
print(json_object['final_response'])
print()
print(f"Input Tokens:  {response.usage.prompt_tokens}")
print(f"Output Tokens: {response.usage.completion_tokens}")
print(f"Cost: ${response.usage.prompt_tokens * 0.15 / 1E6 + response.usage.completion_tokens * 0.6 / 1E6}")
