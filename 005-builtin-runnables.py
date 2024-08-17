import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()

response = chain.invoke("Abram")

print("\n----------\n")

print("Chain with RunnablePassthrough:")

print("\n----------\n")
#print(response)

print("\n----------\n")

def russian_lastname(name: str) -> str:
    return f"{name}ovich"

from langchain_core.runnables import RunnableLambda

chain = RunnablePassthrough() | RunnableLambda(russian_lastname)

response = chain.invoke("Abram")

print("\n----------\n")

print("Chain with RunnableLambda:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "operation_b": RunnableLambda(russian_lastname)
    }
)

response = chain.invoke("Abram")

print("\n----------\n")

print("Chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "soccer_player": lambda x: x["name"]+"ovich"
    }
)

response = chain.invoke({
    "name1": "Jordam",
    "name": "Abram"
})

print("\n----------\n")

print("Chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "soccer_player": RunnableLambda(russian_lastname_from_dictionary),
        "operation_c": RunnablePassthrough(),
    }
) | prompt | model | output_parser

response = chain.invoke({
    "name1": "Jordam",
    "name": "Abram"
})

print("\n----------\n")

print("Chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 10,000 Alumni from all continents and top companies"], embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo")

retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

response = retrieval_chain.invoke("who are the Alumni of AI Accelera?")

print("\n----------\n")

print("Chain with Advanced Use of RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

model = ChatOpenAI(model="gpt-3.5-turbo")

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 5,000 Enterprise Alumni."], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke({"question": "How many Enterprise Alumni has trained AI Accelera?", "language": "Pirate English"})

print("\n----------\n")

print("Chain with RunnableParallel and itemgetter:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain.prompts import PromptTemplate

rock_template = """You are a very smart rock and roll professor. \
You are great at answering questions about rock and roll in a concise\
and easy to understand manner.

Here is a question:
{input}"""

rock_prompt = PromptTemplate.from_template(rock_template)

politics_template = """You are a very good politics professor. \
You are great at answering politics questions..

Here is a question:
{input}"""

politics_prompt = PromptTemplate.from_template(politics_template)

history_template = """You are a very good history teacher. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods.

Here is a question:
{input}"""

history_prompt = PromptTemplate.from_template(history_template)

sports_template = """ You are a sports teacher.\
You are great at answering sports questions.

Here is a question:
{input}"""

sports_prompt = PromptTemplate.from_template(sports_template)

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

general_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
)
prompt_branch = RunnableBranch(
  (lambda x: x["topic"] == "rock", rock_prompt),
  (lambda x: x["topic"] == "politics", politics_prompt),
  (lambda x: x["topic"] == "history", history_prompt),
  (lambda x: x["topic"] == "sports", sports_prompt),
  general_prompt
)

from typing import Literal

from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function


class TopicClassifier(BaseModel):
    "Classify the topic of the user question"
    
    topic: Literal["rock", "politics", "history", "sports"]
    "The topic of the user question. One of 'rock', 'politics', 'history', 'sports' or 'general'."

classifier_function = convert_to_openai_function(TopicClassifier)

llm = ChatOpenAI().bind(functions=[classifier_function], function_call={"name": "TopicClassifier"}) 

parser = PydanticAttrOutputFunctionsParser(pydantic_schema=TopicClassifier, attr_name="topic")

classifier_chain = llm | parser

from operator import itemgetter

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain) 
    | prompt_branch 
    | ChatOpenAI()
    | StrOutputParser()
)

response = final_chain.invoke(
    {"input": "Who was Napoleon Bonaparte?"}
)

print("\n----------\n")

print("Chain with RunnableBranch:")

print("\n----------\n")
print(response)

print("\n----------\n")