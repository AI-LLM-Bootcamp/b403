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
