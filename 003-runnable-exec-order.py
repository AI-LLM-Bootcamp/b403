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

print("LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

prompt.invoke({"soccer_player": "Ronaldo"})

from langchain_core.messages.human import HumanMessage

output_after_first_step = [HumanMessage(content='tell me a curious fact about Ronaldo')]

model.invoke(output_after_first_step)

from langchain_core.messages.ai import AIMessage

output_after_second_step = AIMessage(content='One curious fact about Cristiano Ronaldo is that he does not have any tattoos on his body. Despite the fact that many professional athletes have tattoos, Ronaldo has chosen to keep his body ink-free.', response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 14, 'total_tokens': 52}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c9812511-043a-458a-bfb8-005bc0d057fb-0', usage_metadata={'input_tokens': 14, 'output_tokens': 38, 'total_tokens': 52})

response = output_parser.invoke(output_after_second_step)

print("\n----------\n")

print("After replicating the process without using a LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")