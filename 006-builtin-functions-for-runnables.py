import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("Basic LCEL chain:")

print("\n----------\n")
#print(response)

print("\n----------\n")

chain = prompt | model.bind(stop=["Ronaldo"]) | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("Basic LCEL chain with .bind():")

print("\n----------\n")
#print(response)

print("\n----------\n")

functions = [
    {
      "name": "soccerfacts",
      "description": "Curious facts about a soccer player",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "The question for the curious facts about a soccer player"
          },
          "answer": {
            "type": "string",
            "description": "The answer to the question"
          }
        },
        "required": ["question", "answer"]
      }
    }
  ]

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "soccerfacts"}, functions= functions)
    | JsonOutputFunctionsParser()
)

response = chain.invoke({"soccer_player": "Mbappe"})

print("\n----------\n")

print("Call OpenAI Function in LCEL chain with .bind():")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

def make_uppercase(arg):
    return arg["original_input"].upper()

chain = RunnableParallel({"original_input": RunnablePassthrough()}).assign(uppercase=RunnableLambda(make_uppercase))

response = chain.invoke("whatever")

print("\n----------\n")

print("Basic LCEL chain with .assign():")

print("\n----------\n")
print(response)

print("\n----------\n")