import json


def get_questions(json_data):
    """
    Get a list of Tuples, where each Tuple contains a question and its potential answers
    :param json_data: Json object
    :return: list[Tuple(str, list[str])]
    """
    questions_answers = []
    data = json_data['data']
    for item in data:
        paragraphs = item['paragraphs']
        for paragraph in paragraphs:
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                answers = [a for a in qa['answers']]
                questions_answers.append((question, answers))
    return questions_answers


# load the Json file
with open("../data/squad_dev-v2.0.json", 'r') as jfile:
    jdata = json.load(jfile)


# Grab all questions and answers
all_qas = get_questions(jdata)

# Grab the first 5000
qas_we_want = all_qas[:5000]
