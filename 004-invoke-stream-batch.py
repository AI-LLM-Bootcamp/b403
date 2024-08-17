import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me one sentence about {politician}.")
chain = prompt | model

response = chain.invoke({"politician": "Churchill"})

print("\n----------\n")

print("Response with invoke:")

print("\n----------\n")
print(response.content)

print("\n----------\n")
    
print("\n----------\n")

print("Response with stream:")

print("\n----------\n")
for s in chain.stream({"politician": "F.D. Roosevelt"}):
    print(s.content, end="", flush=True)

print("\n----------\n")

response = chain.batch([{"politician": "Lenin"}, {"politician": "Stalin"}])

print("\n----------\n")

print("Response with batch:")

print("\n----------\n")
print(response)

print("\n----------\n")