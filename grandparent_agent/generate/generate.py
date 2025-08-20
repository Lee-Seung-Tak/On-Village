
from langchain.prompts import ChatPromptTemplate
import boto3
import json
from .util.util import grandparent_agent, output_parser


def llm_generate( user_input: str ): 
    try:
        messages = [
            ("system", "당신은 기쁨이입니다."),
            ("user", "{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | grandparent_agent | output_parser

        response = chain.invoke({ "question": user_input })

        print("Response from model: ", response)

    except Exception as e:
        print(f"ERROR: Can't invoke ==> Reason: {e}")

