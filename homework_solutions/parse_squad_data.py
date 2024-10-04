import json


def get_squad_questions(json_data):
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
                # skip questions that are impossible
                if not qa['is_impossible']:
                    question = qa['question']
                    # collect all answers
                    answers = [a for a in qa['answers']]
                    # append to list of questions
                    questions_answers.append((question, answers))
    return questions_answers


def get_squad_context(json_data):
    """
    Get a list of each context chunk from the dataset
    :param json_data: Json object
    :return: list[str]
    """
    all_context = []
    data = json_data['data']
    for item in data:
        paragraphs = item['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            all_context.append(context)
    return all_context

