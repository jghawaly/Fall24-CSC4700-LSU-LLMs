from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time


# load environmental variables
load_dotenv('../.env')

roman_emperors = [
    "Augustus", "Tiberius", "Caligula", "Claudius", "Nero", "Galba", "Otho", "Vitellius", "Vespasian",
    "Titus", "Domitian", "Nerva", "Trajan", "Hadrian", "Antoninus Pius", "Marcus Aurelius", "Commodus",
    "Pertinax", "Didius Julianus", "Septimius Severus", "Caracalla", "Geta", "Macrinus", "Elagabalus",
    "Severus Alexander", "Maximinus Thrax", "Gordian I", "Gordian II", "Pupienus", "Balbinus", "Philip the Arab",
    "Decius", "Trebonianus Gallus", "Aemilian", "Valerian", "Gallienus", "Claudius Gothicus", "Quintillus",
    "Aurelian", "Tacitus", "Florianus", "Probus", "Carus", "Carinus", "Numerian", "Diocletian", "Maximian",
    "Constantius Chlorus", "Galerius", "Severus II", "Maxentius", "Constantine the Great", "Licinius",
    "Constantine II", "Constantius II", "Julian", "Jovian", "Valentinian I", "Valens", "Gratian", "Valentinian II",
    "Theodosius I", "Arcadius", "Honorius", "Theodosius II", "Marcian", "Leo I", "Leo II", "Zeno", "Basiliscus",
    "Anastasius I", "Justin I", "Justin II", "Tiberius II Constantine", "Maurice", "Phocas", "Heraclius",
    "Constantine III", "Heraklonas", "Constans II", "Constantine IV", "Justinian II", "Leontius", "Tiberius III",
    "Philippikos Bardanes", "Anastasius II", "Theodosius III", "Leo III", "Constantine V", "Leo IV", "Constantine VI",
    "Irene", "Nicephorus I", "Staurakios", "Michael I Rhangabe", "Leo V", "Michael II", "Theophilus", "Michael III",
    "Basil I", "Leo VI", "Alexander", "Constantine VII", "Romanos I", "Romanos II", "Nicephorus II",
    "John I Tzimiskes", "Basil II", "Constantine VIII", "Michael IV", "Michael V", "Zoe", "Theodora",
    "Michael VI", "Isaac I Komnenos", "Alexios I Komnenos", "John II Komnenos", "Manuel I Komnenos",
    "Alexios II Komnenos", "Andronikos I Komnenos", "Isaac II Angelos", "Alexios III Angelos", "Alexios IV Angelos",
    "Alexios V Doukas", "Michael VIII Palaiologos", "Andronikos II Palaiologos", "Michael IX Palaiologos",
    "Andronikos III Palaiologos", "John V Palaiologos", "John VI Kantakouzenos", "Manuel II Palaiologos",
    "John VIII Palaiologos", "Constantine XI Palaiologos"
]

# System and user prompt to be filled in
system_prompt = "You are an expert in Roman history. Your job is to generate a short (150 words maximum) bio for a Roman emperor."
user_prompt = 'The emperor you should consider is {emperor_name}. Return your response in json, with one key, being "response"'

# this list will hold the individual tasks for each sample
tasks = []
for emperor in roman_emperors:
    # created our messages list. The system prompt has not custom fields, but the user prompt has to have emperor_name
    # filled with the current emperor
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(emperor_name=emperor)}]

    # this is a custom id to keep track of each sample (IT MUST BE UNIQUE)
    custom_id = f"emperor={emperor}"

    # this is the actual task to be performed. The response format is json_object, which means our prompt must contain
    # json instructions (as shown above in user prompt)
    task = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": messages
        }
    }
    tasks.append(task)


# Here, we are writing a local file to store the tasks. This is a jsonl file, newline delimited)
with open("../data/input_batch.jsonl", 'w') as jfile:
    for task in tasks:
        jfile.write(json.dumps(task) + '\n')

# establish OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# upload our batch file to OpenAI
batch_file = client.files.create(
    file=open("../data/input_batch.jsonl", 'rb'),
    purpose='batch'
)

# Run the batch using the completions endpoint
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

# loop until the status of our batch is co
# mpleted
complete = False
while not complete:
    check = client.batches.retrieve(batch_job.id)
    print(f'Status: {check.status}')
    if check.status == 'completed':
        complete = True
    time.sleep(1)
print("Done processing batch.")

print("Writing data...")
# Write the results to a local file, again, jsonl format
result = client.files.content(check.output_file_id).content
output_file_name = "../data/output_batch.jsonl"
with open(output_file_name, 'wb') as file:
    file.write(result)

# load the output file, extract each sample output, and and append to a list
results = []
with open(output_file_name, 'r') as file:
    for line in file:
        # this converts the string into a Json object
        json_object = json.loads(line.strip())
        results.append(json_object)

# Show the responses
for item in results:
    print("Model's Response:")
    print('\t', item['response']['body']['choices'][0]['message']['content'])

