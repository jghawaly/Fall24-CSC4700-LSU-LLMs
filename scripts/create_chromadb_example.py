import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# load environmental variables
load_dotenv('../.env')

# Establish client
chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# create a collection named documents, with a provided embedding function
collection = chroma_client.create_collection(
    name="documents",
    embedding_function=openai_ef
)

# create a few documents
documents = [
    {
        'id': 'doc1',
        'text': 'George Washington (February 22, 1732 – December 14, 1799) was an American Founding Father, politician, military officer, and farmer who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted the current Constitution of the United States. Washington has thus become commonly known as the "Father of his Country"',
        'category': 'presidents'
    },
    {
        'id': 'doc2',
        'text': 'John Adams (October 30, 1735 – July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.',
        'category': 'presidents'
    },
    {
        'id': 'doc3',
        'text': 'Æthelstan or Athelstan (/ˈæθəlstæn/; Old English: Æðelstān [ˈæðelstɑːn]; Old Norse: Aðalsteinn; lit. noble stone;[4] c. 894 – 27 October 939) was King of the Anglo-Saxons from 924 to 927 and King of the English from 927 to his death in 939.[a] He was the son of King Edward the Elder and his first wife, Ecgwynn. Modern historians regard him as the first King of England and one of the "greatest Anglo-Saxon kings".[6] He never married and had no children; he was succeeded by his half-brother, Edmund I.',
        'category': 'English kings'
    },
    {
        'id': 'doc4',
        'text': 'Edmund I or Eadmund I[a] (920/921 – 26 May 946) was King of the English from 27 October 939 until his death in 946. He was the elder son of King Edward the Elder and his third wife, Queen Eadgifu, and a grandson of King Alfred the Great. After Edward died in 924, he was succeeded by his eldest son, Edmunds half-brother Æthelstan. Edmund was crowned after Æthelstan died childless in 939. He had two sons, Eadwig and Edgar, by his first wife Ælfgifu, and none by his second wife Æthelflæd. His sons were young children when he was killed in a brawl with an outlaw at Pucklechurch in Gloucestershire, and he was succeeded by his younger brother Eadred, who died in 955 and was followed by Edmunds sons in succession.',
        'category': 'English kings'
    },
]

ids = [doc['id'] for doc in documents]
texts = [doc['text'] for doc in documents]
metadatas = [{'category': doc['category']} for doc in documents]

collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas
)
