import chromadb
from langchain.vectorstores import Chroma

# ChromaDB 클라이언트 연결
client = chromadb.HttpClient(host="chroma", port=8000)

# 'prompt_collection' 컬렉션 생성 또는 가져오기
collection = client.get_or_create_collection(name="prompt_collection")

# 데이터 정의
prompts = [
    {
        "intent": "자기소개",
        "description": "AI 자신을 소개하는 요청",
        "examples": ["기쁨이 서비스가 뭐야?","자기소개 해줘","넌 뭘 할 수 있어?","너의 기능이 뭐야"],
        "response_type": "system_info",
        "sample_prompt": """저는 어르신들께 소소한 기쁨을 드리고자 만들어진 인공지능 에이전트 기쁨이에요. 저는 어르신들의 하루의 소중한 일상에 대해서
        이야기 나누는것을 좋아해요. 그리고 어르신들의 약 시간을 알려드릴 수 있어요. 그리고 재미있는 이야기도 들려드릴 수 있고, 어르신들께서 마을에서 
        생활하다 발견하는 불편사항을 저한테 말씀해주시면 정리해서 시에 전달할 수 있어요."""
    },
]


import boto3
import numpy as np
import json
# AWS Bedrock Embeddings (예시: text-embedding-3-small)
bedrock = boto3.client("bedrock", region_name="ap-northeast-2")

from langchain.embeddings.base import Embeddings

class AWSEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = bedrock.invoke_model(
            modelId="anthropic.claude-embedding-3-small",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"input": text})
        )
        result = json.loads(response["body"])
        return np.array(result["embedding"])


from langchain.prompts import ChatPromptTemplate

vectorstore = Chroma(
    client=client,
    collection_name="prompt_collection",
    embedding_function=AWSEmbeddings
)

top_k = 3
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

# 2) LLM + Chain 세팅 (grandparent_agent + output_parser 사용)

# ChatPromptTemplate 예시
system_template = """
    당신은 AI assistant '기쁨이'입니다.
    어르신을 존중하며 존댓말로만 대답하세요.
    아래는 참고할 수 있는 프롬프트입니다:
    {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{question}")
])

# chromadb에 prompts 저장
def insert_chromadb() :
    # ChromaDB에 넣을 수 있도록 데이터 포맷 변경
    ids = [f"doc_{i}" for i in range(len(prompts))]
    documents = [p["sample_prompt"] for p in prompts]
    metadatas = [
        {
            "intent": p["intent"], 
            "description": p["description"], 
            "response_type": p["response_type"],
            "examples": ", ".join(p["examples"])
        } for p in prompts
    ]

    # 컬렉션에 데이터 추가
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"{len(prompts)}개의 문서가 성공적으로 벡터 DB에 추가되었습니다.")

    # 데이터가 잘 들어갔는지 확인 (선택 사항)
    results = collection.get(ids=ids)
    print("\n저장된 문서 확인:")
    print(results, flush=True)