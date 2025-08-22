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


import boto3, json

# Bedrock runtime client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-embed-text-v2:0"



def titan_embed(text: str):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(modelId=model_id, body=body)
    model_response = json.loads(response["body"].read())
    return model_response["embedding"]  # 1024차원 리스트 반환

def insert_chromadb():
    client.delete_collection("prompt_collection")

    # Titan 임베딩(1024차원)에 맞는 새 컬렉션 생성
    collection = client.create_collection(
        name="prompt_collection",
        metadata={"hnsw:space": "cosine"}  # 보통 cosine distance 사용
    )
    
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

    # Titan 임베딩 직접 생성
    embeddings = [titan_embed(doc) for doc in documents]

    # ChromaDB에 Titan 임베딩과 함께 저장
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"{len(prompts)}개의 문서가 Titan 임베딩으로 성공적으로 벡터 DB에 추가되었습니다.")