import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Write one brief sentence about {politician}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({
    "politician": "JFK"
})

print("\n----------\n")

print("Basic LCEL chain with output parser:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

vectorstore = DocArrayInMemorySearch.from_texts(
    ["AI Accelera has provided Generative AI Training and Consulting Services in more than 100 countries", "Aceleradora AI is the branch of AI Accelera for the Spanish-Speaking market"],
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

get_question_and_retrieve_relevant_docs = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = get_question_and_retrieve_relevant_docs | prompt | model | output_parser

response = chain.invoke("In how many countries has AI Accelera provided services?")

print("\n----------\n")

print("Mid-level LCEL chain with retriever:")

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

print("Mid-level LCEL chain with RunnableParallel:")

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
    ["AI Accelera has trained more than 3,000 Enterprise Alumni."], embedding=OpenAIEmbeddings()
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

print("Mid-level LCEL chain with RunnableParallel and itemgetter:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    user_input = RunnablePassthrough(),
    transformed_output = lambda x: x["num"] + 1,
)

response = runnable.invoke({"num": 1})

print("\n----------\n")

print("Mid-level LCEL chain with RunnablePassthrough:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a sentence about {politician}")

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | model | StrOutputParser()

response = chain.invoke("Chamberlain")

print("\n----------\n")

print("Chaining Runnables:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.output_parsers import StrOutputParser

historian_prompt = ChatPromptTemplate.from_template("Was {politician} positive for Humanity?")

composed_chain = {"politician": chain} | historian_prompt | model | StrOutputParser()

response = composed_chain.invoke({"politician": "Lincoln"})

print("\n----------\n")

print("Coercion:")

print("\n----------\n")
#print(response)

print("\n----------\n")

composed_chain_with_lambda = (
    chain
    | (lambda input: {"politician": input})
    | historian_prompt
    | model
    | StrOutputParser()
)

response = composed_chain_with_lambda.invoke({"politician": "Robespierre"})

print("\n----------\n")

print("Chain with lambda function:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt1 = ChatPromptTemplate.from_template("what is the country {politician} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what continent is the country {country} in? respond in {language}"
)

model = ChatOpenAI()

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"country": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

response = chain2.invoke({"politician": "Miterrand", "language": "French"})

print("\n----------\n")

print("Multiple Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"

chain = RunnableParallel(
    {
        "soccer_player": RunnablePassthrough() 
        | RunnableLambda(russian_lastname_from_dictionary), 
        "operation_c": RunnablePassthrough()
    }
) | prompt | model | output_parser

response = chain.invoke({
    "name1": "Jordam",
    "name": "Abram"
})

print("\n----------\n")

print("Nested Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")

# First let's create a chain with a ChatModel
# We add in a string output parser here so the outputs between the two are the same type
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a funny assistant who always includes a joke in your response",
        ),
        ("human", "Who is the best {sport} player worldwide?"),
    ]
)
# Here we're going to use a bad model name to easily create a chain that will error
chat_model = ChatOpenAI(model="gpt-fake")

bad_chain = chat_prompt | chat_model | StrOutputParser()

# Now lets create a chain with the normal OpenAI model
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt_template = """Instructions: You're a funny assistant who always includes a joke in your response.

Question: Who is the best {sport} player worldwide?"""

prompt = PromptTemplate.from_template(prompt_template)

llm = OpenAI()

good_chain = prompt | llm

# We can now create a final chain which combines the two
chain = bad_chain.with_fallbacks([good_chain])

response = chain.invoke({"sport": "soccer"})

print("\n----------\n")

print("Fallback for Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")