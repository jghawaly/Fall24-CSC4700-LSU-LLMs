import json
import time
import os
from parse_squad_data import get_squad_questions, get_squad_context
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm
import argparse
import chromadb
from chromadb.utils import embedding_functions


def format_answers_list(answers: list[str]) -> str:
    """
    Generated a string version of a list of answers
    :param answers: list of answers, list[str]
    :return: pretty string version of the list of answers, str
    """
    formatted_answers = ""
    for i in range(len(answers)):
        formatted_answers += f"\t{i}. {answers[i]['text']}"
    return formatted_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run GPT-4o-mini and Llama 3.1 8B on Squad v2 dev set. Uses GPT-4o-mini for grading")
    parser.add_argument('--create_db', action='store_true', help="creates the Chroma database at the provided path")
    parser.add_argument('--db_path', type=str, default='../data/squad_db',
                        help='path to where the database will be stored')
    parser.add_argument('--run_openai', action='store_true', help="runs GPT-4o-mini on the questions")
    parser.add_argument('--run_llama', action='store_true', help="runs Llama 3.1 8B on the questions")
    parser.add_argument('--run_grader', action='store_true', help="runs GPT-4o to grade question responses")
    parser.add_argument('--view_scores', action='store_true', help="view the scores for each model")
    parser.add_argument('--n', type=int, default=5000, help='number of samples to process')
    parser.add_argument('--k', type=int, default=5, help='number of context chunks to retrieve')
    args = parser.parse_args()

    # load environmental variables
    load_dotenv('../.env')

    # load the Json file we downloaded for the Squad v2 dataset
    with open("../data/squad_dev-v2.0.json", 'r') as jfile:
        jdata = json.load(jfile)

    # Grab all questions and answers
    all_qas = get_squad_questions(jdata)

    # Grab all context chunks
    all_chunks = get_squad_context(jdata)

    # Grab the first 5000 questions
    qas_we_want = all_qas[:args.n]

    # Set our system prompt for the Q/A
    qa_system_prompt = """You are an AI who is tasked with concisely answering questions/trivia.

    Guidelines:
    1. You must only use the provided context to answer the question.
    2. You must not produce any information that is not in the provided context.
    3. If the provided context does not contain the answer, you should respond with "I am sorry, but I can't answer the question."
    4. Only answer the question, don't add additional bloat content. Your answer should be concise.

    Here are the context chunks. Be aware that some (or even all) chunks may not be relevant to the question:
    {context}

    Here is the question: {question}

    Your response: """

    # Set our system prompt for the GPT-4o grader
    grader_system_prompt = """You are a teacher tasked with determining whether a student’s answer to a question was correct, based on a set of possible correct answers. You must only use the provided possible correct answers to determine if the student’s response was correct.

    Question: {question}

    Student’s Response: {student_response}

    Possible Correct Answers:
    {correct_answers}

    Your response should only be a valid Json as shown below:
    {{
    	"explanation" (str): A short explanation of why the student’s answer was correct or incorrect.,
    	"score" (bool): true if the student’s answer was correct, false if it was incorrect
    }}

    Your response: """

    # Create our OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create our Azure Open Model client
    azure_client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )

    # establish ChromaDB client
    chroma_client = chromadb.PersistentClient(path=args.db_path)

    # setup embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    if args.create_db:
        # create a collection named documents, with a provided embedding function
        collection = chroma_client.create_collection(
            name="documents",
            embedding_function=openai_ef
        )

        # generate unique ids for all chunks
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]

        # add the chunks
        collection.add(
            ids=ids,
            documents=all_chunks
        )

    if args.run_openai:
        # BATCH API with OpenAI GPT-4o-mini ----------------------------------------------------------------------------
        # get the documents collection, with a provided embedding function
        collection = chroma_client.get_collection(
            name="documents",
            embedding_function=openai_ef
        )

        # this list will hold the individual tasks for each question
        tasks = []
        i = 0
        for q, _ in tqdm(qas_we_want):
            # retrieve context using embeddings
            retrieved_context = collection.query(query_texts=[q], n_results=args.k)
            formatted_context = "Chunk\n" + "\n\nChunk\n".join([doc[0] for doc in retrieved_context['documents']])

            # created our messages list.
            messages = [{"role": "system", "content": qa_system_prompt.format(question=q, context=formatted_context)}, ]

            # this is a custom id to keep track of each sample (IT MUST BE UNIQUE)
            custom_id = f"openai-question-{i}"

            # this is the actual task to be performed.
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "messages": messages
                }
            }
            tasks.append(task)
            i += 1

        # Here, we are writing a local file to store the tasks. This is a jsonl file, newline delimited)
        with open("../data/squadv2_dev_gpt-4o-mini_RAG_answers_input_batch.jsonl", 'w') as jfile:
            for task in tasks:
                jfile.write(json.dumps(task) + '\n')

        # upload our batch file to OpenAI
        batch_file = openai_client.files.create(
            file=open("../data/squadv2_dev_gpt-4o-mini_RAG_answers_input_batch.jsonl", 'rb'),
            purpose='batch'
        )

        # Run the batch using the completions endpoint
        batch_job = openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        # loop until the status of our batch is completed
        complete = False
        while not complete:
            check = openai_client.batches.retrieve(batch_job.id)
            print(f'Status: {check.status}')
            if check.status == 'completed':
                complete = True
            time.sleep(1)
        print("Done processing batch.")

        print("Writing data...")
        # Write the results to a local file, again, jsonl format
        result = openai_client.files.content(check.output_file_id).content
        output_file_name = "../data/squadv2_dev_gpt-4o-mini_RAG_answers_output_batch.jsonl"
        with open(output_file_name, 'wb') as file:
            file.write(result)

    if args.run_llama:
        # RUN LLaMa-3.1-8B ---------------------------------------------------------------------------------------------
        # get the documents collection, with a provided embedding function
        collection = chroma_client.get_collection(
            name="documents",
            embedding_function=openai_ef
        )

        # this will contain the model's response objects for each question
        response_objects = []
        i = 0
        for q, _ in tqdm(qas_we_want):
            custom_id = f"llama-question-{i}"

            # retrieve context using embeddings
            retrieved_context = collection.query(query_texts=[q], n_results=args.k)
            formatted_context = "Chunk\n" + "\n\nChunk\n".join([doc[0] for doc in retrieved_context['documents']])

            # send the question to the model and get a response
            response = azure_client.complete(
                messages=[
                    SystemMessage(content=qa_system_prompt.format(question=q, context=formatted_context))
                ]
            )
            # extract the response
            response_objects.append(
                {"custom_id": custom_id, "response": json.dumps(response['choices'][0]['message']['content'])})
            i += 1

        with open("../data/squadv2_dev_llama3-1-8b_RAG_answers.jsonl", 'w') as jfile:
            for r in response_objects:
                jfile.write(json.dumps(r) + '\n')

    if args.run_grader:
        # BATCH API with OpenAI GPT-4o ---------------------------------------------------------------------------------
        # load the model responses
        all_model_responses = {}


        def load_and_store(f_path):
            """
            This function loads the saved responses and stores them in all_model_responses
            :param f_path: path to saved jsonl file
            :return: None
            """
            with open(f_path, 'r') as file:
                for line in file:
                    # this converts the string into a Json object
                    json_object = json.loads(line.strip())
                    # openai output will be a bit differently formatted than llama
                    if 'body' in json_object['response']:
                        all_model_responses[json_object['custom_id']] = \
                        json_object['response']['body']['choices'][0]['message']['content']
                    else:
                        all_model_responses[json_object['custom_id']] = json_object['response']


        def generate_task(question: str,
                          answers: list[str],
                          custom_id: str,
                          model_response: str) -> dict:
            """
            Generate a grader task, formatted to use OpenAI structured output
            :param question: the question
            :param answers: list of possible answers
            :param custom_id: a custom id for this task (must be unique)
            :param model_response: the model's response to be graded
            :return: a task that can be used with the OpenAI batch API
            """
            messages = [{"role": "system",
                         "content": grader_system_prompt.format(question=question,
                                                                student_response=model_response,
                                                                correct_answers=format_answers_list(answers))}]

            # this is the actual task to be performed.
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    'model': 'gpt-4o-mini-2024-07-18',
                    'messages': messages,
                    'temperature': 0.0,
                    'response_format': {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "grade_determination",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "explanation": {
                                        "type": "string",
                                        "description": "A concise explanation of how you chose the score"
                                    },
                                    "score": {"type": "boolean",
                                              "description": "true if the student was correct, false otherwise"}
                                },
                                "required": ["explanation", "score"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                }
            }
            return task


        load_and_store("../data/squadv2_dev_gpt-4o-mini_RAG_answers_output_batch.jsonl")
        load_and_store("../data/squadv2_dev_llama3-1-8b_RAG_answers.jsonl")

        # this list will hold the individual tasks for each question
        tasks = []
        i = 0
        for q, answers in tqdm(qas_we_want):
            # for each question, we create a task for the llama response and one for the gpt-4o-mini response
            openai_id = f"openai-question-{i}"
            llama_id = f"llama-question-{i}"

            # get the responses for the OpenAI and Llama models
            openai_response = all_model_responses[openai_id]
            llama_response = all_model_responses[llama_id]

            # create grading tasks for both models
            tasks.append(generate_task(q, answers, openai_id, openai_response))
            tasks.append(generate_task(q, answers, llama_id, llama_response))
            i += 1

        # Here, we are writing a local file to store the tasks. This is a jsonl file, newline delimited)
        with open("../data/squadv2_dev_RAG_scoring_input_batch.jsonl", 'w') as jfile:
            for task in tasks:
                jfile.write(json.dumps(task) + '\n')

        # upload our batch file to OpenAI
        batch_file = openai_client.files.create(
            file=open("../data/squadv2_dev_RAG_scoring_input_batch.jsonl", 'rb'),
            purpose='batch'
        )

        # Run the batch using the completions endpoint
        batch_job = openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        # loop until the status of our batch is completed
        complete = False
        while not complete:
            check = openai_client.batches.retrieve(batch_job.id)
            print(f'Status: {check.status}')
            if check.status == 'completed':
                complete = True
            time.sleep(1)
        print("Done processing batch.")

        print("Writing data...")
        # Write the results to a local file, again, jsonl format
        result = openai_client.files.content(check.output_file_id).content
        output_file_name = "../data/squadv2_dev_RAG_scoring_output_batch.jsonl"
        with open(output_file_name, 'wb') as file:
            file.write(result)

    if args.view_scores:
        llama_total_count = 0
        llama_correct_count = 0
        openai_total_count = 0
        openai_correct_count = 0
        with open('../data/squadv2_dev_RAG_scoring_output_batch.jsonl', 'rb') as jfile:
            for line in jfile:
                # this converts the string into a Json object
                json_object = json.loads(line.strip())
                custom_id = json_object["custom_id"]
                result_json = json.loads(json_object['response']['body']['choices'][0]['message']['content'])
                if 'llama' in custom_id:
                    llama_total_count += 1
                    if result_json["score"]:
                        llama_correct_count += 1
                if 'openai' in custom_id:
                    openai_total_count += 1
                    if result_json["score"]:
                        openai_correct_count += 1

        # show the results
        print(
            f"RAG Llama 3.1 8B: {llama_correct_count} out of {llama_total_count}, or {llama_correct_count / llama_total_count:.2f}, questions answered correctly")
        print(
            f"RAG GPT-4o-mini:  {openai_correct_count} out of {openai_total_count}, or {openai_correct_count / openai_total_count:.2f}, questions answered correctly")
