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

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

prompt.invoke({"soccer_player": "Ronaldo"})

from langchain_core.messages.human import HumanMessage

output_after_first_step = [HumanMessage(content='tell me a curious fact about Ronaldo')]

model.invoke(output_after_first_step)

from langchain_core.messages.ai import AIMessage

# This AIMessage is a saved example of the model's answer, taken when the
# course was recorded, so its metadata still shows the model used then
# (gpt-3.5-turbo-0125). It does not call any model: it only shows what the
# output of the second step looks like. If you print the result of
# model.invoke(output_after_first_step), you will see the same structure,
# with gpt-6-luna as model_name.
output_after_second_step = AIMessage(content='One curious fact about Cristiano Ronaldo is that he does not have any tattoos on his body. Despite the fact that many professional athletes have tattoos, Ronaldo has chosen to keep his body ink-free.', response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 14, 'total_tokens': 52}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c9812511-043a-458a-bfb8-005bc0d057fb-0', usage_metadata={'input_tokens': 14, 'output_tokens': 38, 'total_tokens': 52})

response = output_parser.invoke(output_after_second_step)

print("\n----------\n")

print("After replicating the process without using a LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")
