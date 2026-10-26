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
chatModel = ChatOpenAI(model=MODEL, temperature=1)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {politician}")

# The chain works exactly as before: StrOutputParser turns the model's
# message into plain text, so response is already a string.
chain = prompt | chatModel | StrOutputParser()

response = chain.invoke({"politician": "JFK"})

print("\n----------\n")

print("Result from invoking the chain:")

print("\n----------\n")
print(response)

print("\n----------\n")
