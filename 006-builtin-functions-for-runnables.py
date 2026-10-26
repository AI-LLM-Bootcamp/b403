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
#
# Further down, this same model calls a function (soccerfacts). Recent OpenAI
# models only accept functions and tools through this endpoint when reasoning
# is turned off, so we set reasoning_effort to "none". The model then answers
# directly, as the original model did. This version of langchain-openai does
# not have a reasoning_effort option yet, so we send it through extra_body,
# which passes it straight to OpenAI.
model = ChatOpenAI(
    model=MODEL,
    temperature=1,
    extra_body={"reasoning_effort": "none"}
)

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

# Original version: this example used .bind(stop=["Ronaldo"]), which made the
# model stop writing as soon as it produced the word "Ronaldo". That worked
# with the OpenAI models available when the course was recorded, but recent
# OpenAI models no longer accept the stop parameter.
# The idea of .bind() is the same: it fixes an argument of the model inside
# the chain. Here we fix max_completion_tokens, the maximum length of the
# answer, so you will see the answer cut off after about 20 tokens.
# This version of the openai library does not know max_completion_tokens
# yet, so we send it through extra_body, like reasoning_effort above.
# The extra_body we bind here replaces the model's own extra_body, so we
# repeat reasoning_effort="none" next to the new limit.
chain = prompt | model.bind(
    extra_body={"reasoning_effort": "none", "max_completion_tokens": 20}
) | output_parser

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
