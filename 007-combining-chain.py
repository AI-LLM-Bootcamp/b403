import os
from dotenv import load_dotenv, find_dotenv
# Remember: the .env file in the project folder must contain the line
# OPENAI_API_KEY=sk-... (no quotes, no spaces).
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

# Original version: this example used gpt-3.5-turbo, which was a good
# choice when the course was recorded. OpenAI has since retired it.
# Today we use a current chat model: small, fast and affordable for practice.
# It is defined once here and reused in the rest of the file.
MODEL = "gpt-6-luna"

# This version of langchain-openai sends temperature=0.7 by default,
# but recent OpenAI models only accept the default value (1).
# That is why every model in this file sets temperature=1 explicitly.
model = ChatOpenAI(model=MODEL, temperature=1)

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

# Original version: OpenAIEmbeddings() used its default model,
# text-embedding-ada-002. That model still works, but OpenAI now presents
# text-embedding-3-small as its current small embedding model: it performs
# a bit better and costs less. Its vectors have the same length (1536).
# Writing the model name makes it easy to see and change. It is reused in
# the other vector stores of this file.
EMBEDDING_MODEL = "text-embedding-3-small"

vectorstore = DocArrayInMemorySearch.from_texts(
    ["AI Accelera has provided Generative AI Training and Consulting Services in more than 100 countries", "Aceleradora AI is the branch of AI Accelera for the Spanish-Speaking market"],
    embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
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
    ["AI Accelera has trained more than 10,000 Alumni from all continents and top companies"], embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL)
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Original version: gpt-3.5-turbo, now retired. We reuse MODEL.
model = ChatOpenAI(model=MODEL, temperature=1)

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

# Original version: gpt-3.5-turbo, now retired. We reuse MODEL.
model = ChatOpenAI(model=MODEL, temperature=1)

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 3,000 Enterprise Alumni."], embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL)
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

# Original version: gpt-3.5-turbo, now retired. We reuse MODEL.
model = ChatOpenAI(model=MODEL, temperature=1)

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

# Original version: ChatOpenAI() used the library's default model,
# gpt-3.5-turbo, which OpenAI has since retired. We reuse MODEL.
model = ChatOpenAI(model=MODEL, temperature=1)

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
from langchain_openai import ChatOpenAI

prompt_template = """Instructions: You're a funny assistant who always includes a joke in your response.

Question: Who is the best {sport} player worldwide?"""

prompt = PromptTemplate.from_template(prompt_template)

# Original version: OpenAI() used the completions model
# gpt-3.5-turbo-instruct, which OpenAI has since retired. Current OpenAI
# models are chat models, so we use ChatOpenAI with MODEL. We keep the
# name llm so the rest of the example works the same way.
llm = ChatOpenAI(model=MODEL, temperature=1)

# A chat model returns a message, not plain text like the old completions
# model did. We add StrOutputParser so both chains return text, as the
# comment at the start of this example explains.
good_chain = prompt | llm | StrOutputParser()

# We can now create a final chain which combines the two
chain = bad_chain.with_fallbacks([good_chain])

response = chain.invoke({"sport": "soccer"})

print("\n----------\n")

print("Fallback for Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")
