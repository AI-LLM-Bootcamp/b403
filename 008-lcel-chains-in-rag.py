import os
from dotenv import load_dotenv, find_dotenv
# Remember: the .env file in the project folder must contain the line
# OPENAI_API_KEY=sk-... (no quotes, no spaces).
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

# Original version: this example used gpt-3.5-turbo-0125, which was a good
# choice when the course was recorded. OpenAI has since retired it.
# Today we use a current chat model: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# This version of langchain-openai sends temperature=0.7 by default,
# but recent OpenAI models only accept the default value (1).
# That is why we set temperature=1 explicitly.
model = ChatOpenAI(model=MODEL, temperature=1)

# WebBaseLoader downloads a web page. This version of langchain-community
# shows a warning if your requests do not identify themselves with a
# User-Agent. We set one here, before importing the loader, so the warning
# does not appear. You can change it to any name that identifies your app.
os.environ["USER_AGENT"] = "llm-apps-bootcamp-exercise/1.0"

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

# Original version: OpenAIEmbeddings() used its default model,
# text-embedding-ada-002. That model still works, but OpenAI now presents
# text-embedding-3-small as its current small embedding model: it performs
# a bit better and costs less. Its vectors have the same length (1536).
# Writing the model name makes it easy to see and change.
EMBEDDING_MODEL = "text-embedding-3-small"

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL))

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")

print("\n----------\n")

print("Chains in RAG:")

print("\n----------\n")
print(response)

print("\n----------\n")
