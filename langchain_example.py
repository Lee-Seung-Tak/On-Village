import boto3
import json
from dotenv import load_dotenv
import os
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()

AWS_ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY")
OpenWeather_KEY = os.getenv("WEATHER_KEY")
AWS_REGION = "us-east-1"  # 서울 리전
# 환경 변수 로드
load_dotenv()

# Bedrock 연결
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1',
)

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
)

# 모델 ID 및 설정
model_id = 'amazon.titan-embed-text-v1'
accept = 'application/json'
content_type = 'application/json'

llm_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
# 프롬프트 데이터
prompts = [
    {
        "intent": "자기소개 요청",
        "description": "AI 자신을 소개하는 요청",
        "examples": ["기쁨이 서비스가 뭐야?", "자기소개 해줘", "넌 뭘 할 수 있어?", "너의 기능이 뭐야"],
        "sample_prompt": """저는 어르신들께 소소한 기쁨을 드리고자 만들어진 인공지능 에이전트 기쁨이에요. 저는 어르신들의 하루의 소중한 일상에 대해서
        이야기 나누는 것을 좋아해요. 그리고 어르신들의 약 시간을 알려드릴 수 있어요. 그리고 재미있는 이야기도 들려드릴 수 있고, 어르신들께서 마을에서 
        생활하다 발견하는 불편사항을 저한테 말씀해주시면 정리해서 시에 전달할 수 있어요."""
    },

    {
        "intent": "정보 검색",
        "description": "어떤 정보에 대한 질문",
        "examples": ["김치찌개 끓이는 법", "음식 레시피 알려줘", "약국 전화번호 알려줘", "병원 몇시까지 하는지 알려줘"],
        "sample_prompt": """사용자의 질문을 토대로 구글에 검색을 하고, 해당 결과를 바탕으로 답변합니다."""
    },

    {
        "intent": "날씨 질문",
        "description": "날씨 질문",
        "examples": ["오늘 날씨 어때", "서울 날씨 어때", "의정부 날씨 어때"],
        "sample_prompt": """제공된 날씨 데이터를 기반으로 날씨 정보를 답변합니다."""
    },

    {
        "intent": "긴급 신고",
        "description": "사용자의 응급 상황 판단",
        "examples": ["도와줘", "살려줘", "응급 상황이야"],
        "sample_prompt": """긴급 상황이 감지되었습니다. 즉시 119에 신고 절차를 진행합니다."""
    },
]

# 코사인 유사도 함수
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 임베딩 생성
embeddings = []

for p in prompts:
    for ex in p["examples"]:
        body = json.dumps({"inputText": ex})

        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')

        embeddings.append({
            "intent": p["intent"],
            "sample_prompt": p["sample_prompt"],
            "text": ex,
            "embedding": embedding
        })

# === 사용자 입력 ===
user_input = "자기소개 해봐"

# 사용자 입력 임베딩
body = json.dumps({"inputText": user_input})
response = bedrock_runtime.invoke_model(
    body=body,
    modelId=model_id,
    accept=accept,
    contentType=content_type
)
response_body = json.loads(response['body'].read())
user_embedding = response_body.get('embedding')

# 가장 가까운 intent 찾기
best_match = None
best_score = -1

for item in embeddings:
    score = cosine_similarity(user_embedding, item["embedding"])
    if score > best_score:
        best_score = score
        best_match = item

# 결과 출력
print("사용자 입력:", user_input)
print("가장 가까운 Intent:", best_match["intent"])
print("매칭된 Example:", best_match["text"])
print("추천할 Sample Prompt:", best_match["sample_prompt"])
print("유사도 점수:", best_score)

import chromadb
from chromadb.config import Settings

# REST API 서버에 붙기 (Docker에서 -p 8000:8000 으로 열려있다고 가정)
chroma_client = chromadb.HttpClient(
    host="localhost",  # Docker를 로컬에서 띄운 경우
    port=8003
)

collection = chroma_client.get_or_create_collection(name="prompts")

for i, item in enumerate(embeddings):
    collection.add(
        ids=[f"prompt_{i}"],  # 유니크 ID
        embeddings=[item["embedding"]],  # 벡터
        documents=[item["text"]],  # 원문 (example 문장)
        metadatas=[{
            "intent": item["intent"],
            "sample_prompt": item["sample_prompt"]
        }]
    )

# 사용자 입력
user_input = "자기소개 해봐"

# 임베딩 생성
body = json.dumps({"inputText": user_input})
response = bedrock_runtime.invoke_model(
    body=body,
    modelId=model_id,
    accept=accept,
    contentType=content_type
)
response_body = json.loads(response['body'].read())
user_embedding = response_body.get('embedding')

# ChromaDB에서 유사도 검색
results = collection.query(
    query_embeddings=[user_embedding],
    n_results=1
)

print(results)




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

# Retriever 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})



from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock
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

# Tool 래퍼
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
        return "error"




llm_weather_tool_wrapped = Tool(
    name="Weather Tool",
    func=llm_weather_tool,
    description="사용자 위치를 받아 날씨 정보를 제공합니다."
)

tools = [retriever_tool, llm_weather_tool_wrapped]  # Tool 객체만 넣기


# Agent 생성
agent = initialize_agent(
    tools=tools,
    llm=grandparent_agent,
    agent="zero-shot-react-description",
    verbose=True
)


# output = agent(query_input)
query_input = '김치 담구는 법 알려줘 | location: {"lat":37.5665,"lon":126.9780}'
output = agent(query_input)
print(output)





# import json
# import requests
# print(OpenWeather_KEY)
# def llm_weather_tool(location):
#     location = json.loads(location)
#     lat = location['lat']
#     lon = location['lon']
#     api_url = (
#         f"https://api.openweathermap.org/data/2.5/weather?"
#         f"lat={lat}&lon={lon}&appid={OpenWeather_KEY}&units=metric"
#     )
#     response = requests.get(api_url)
#     print(response.json())
#     if response.status_code == 200:
#         return json.dumps(response.json())
#     else:
#         return "error"
    
# llm_weather_tool('{"lat":37.5665,"lon":126.9780}')