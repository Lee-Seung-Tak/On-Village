
from langchain.prompts import ChatPromptTemplate
import boto3
import json
from .tts import text_to_speech_aws_polly

import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()

###########################################################################################################
# 사전 정의
###########################################################################################################
AWS_ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY")
OpenWeather_KEY = os.getenv("WEATHER_KEY")

AWS_REGION      = "us-east-1"  # 서울 리전
model_id        = 'amazon.titan-embed-text-v1'
accept          = 'application/json'
content_type    = 'application/json'
llm_model_id    = "anthropic.claude-3-haiku-20240307-v1:0"

# Bedrock 연결
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1',
)

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
)
###########################################################################################################



###########################################################################################################
# chrom db 관련
###########################################################################################################
import chromadb
from chromadb.config import Settings

# chroma client init
chroma_client = chromadb.HttpClient(
    host="on-village_chroma",  # Docker를 로컬에서 띄운 경우
    port=8000
)

# select collection
collection = chroma_client.get_or_create_collection(name="prompts")
###########################################################################################################


###########################################################################################################
# 임베딩 관련
###########################################################################################################
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# Bedrock 임베딩을 LangChain용으로 래핑
class BedrockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            body = json.dumps({"inputText": text})
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response['body'].read())
            embeddings.append(response_body["embedding"])
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# 기존 ChromaDB 컬렉션 연결
vectorstore = Chroma(
    client=chroma_client,
    collection_name="prompts",
    embedding_function=BedrockEmbeddings()
)

# retriever 정의
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

###########################################################################################################



###########################################################################################################
# llm 관련
###########################################################################################################
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock
from langchain.schema import StrOutputParser
import boto3

aws_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

model_kwargs = {
    "max_tokens": 5000,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"]
}

grandparent_agent = ChatBedrock(
    region_name=AWS_REGION,
    client=aws_client,
    model_id=llm_model_id,
    model_kwargs=model_kwargs
)

# RetrievalQA 체인 정의
retrieval_chain = RetrievalQA.from_chain_type(
    llm=grandparent_agent,
    retriever=retriever,
    return_source_documents=False
)
###########################################################################################################



###########################################################################################################
# retriver tool 정의
###########################################################################################################
def retriever_with_sample_prompt(query: str) -> str:
    # 검색 결과 가져오기
    retrieved = retriever.get_relevant_documents(query)
    # sample_prompt만 뽑아서 LLM 입력에 포함
    sample_prompt_text = retrieved[0].metadata.get("sample_prompt", "")
    # LLM에게 질문과 sample_prompt를 같이 전달
    return f"{sample_prompt_text}\n\n질문: {query}"

retriever_tool = Tool(
    name="prompt_retriever",
    func=retriever_with_sample_prompt,
    description="사용자 질문과 유사한 프롬프트를 검색해서 답변합니다."
)
###########################################################################################################



###########################################################################################################
# 날씨 tool 정의
###########################################################################################################
import json
import requests
def llm_weather_tool(location):
    location = json.loads(location)
    lat = location['lat']
    lon = location['lon']

    api_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={OpenWeather_KEY}&units=metric"
    )
 
    response = requests.get(api_url)
    if response.status_code == 200:
        return json.dumps(response.json())
    else:
        return "날씨 툴 call 호출에 실패했습니다!!!!"


llm_weather_tool_wrapped = Tool(
    name="Weather Tool",
    func=llm_weather_tool,
    description="사용자 위치를 받아 날씨 정보를 제공합니다. 반드시 사용자가 질문한 언어로 답변합니다."
)
###########################################################################################################

tools = [retriever_tool, llm_weather_tool_wrapped]  # Tool 객체만 넣기


# Agent 생성
agent = initialize_agent(
    tools=tools,
    llm=grandparent_agent,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,  # 포맷 오류가 나도 그냥 진행
    # max_iterations=5             # 5번만 반복
)


async def rag_to_speech(user_input: str, user_information, voice_id="Seoyeon", engine="neural"):

    print("user_input : ", user_input, flush=True)
    llm_response = agent(user_input + user_information)
    print("llm_response : ", llm_response['output'], flush=True)
    # TTS 변환
    audio_bytes = await text_to_speech_aws_polly(llm_response['output'])

    return {
        "text": llm_response['output'],
        "audio_bytes": audio_bytes
    }
