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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

# Both chains below use the same model, so the comparison between the
# legacy chain and the LCEL chain works exactly as before.
traditional_chain = LLMChain(
    llm=model,
    prompt=prompt
)

response = traditional_chain.predict(soccer_player="Maradona")

print("\n----------\n")

print("Legacy chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")
