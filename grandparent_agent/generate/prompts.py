import chromadb
from langchain_chroma import Chroma   

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


# import boto3, json

# # Bedrock runtime client
# bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
# model_id = "amazon.titan-embed-text-v2:0"



# def titan_embed(text: str):
#     body = json.dumps({"inputText": text})
#     response = bedrock.invoke_model(modelId=model_id, body=body)
#     model_response = json.loads(response["body"].read())
#     return model_response["embedding"]  # 1024차원 리스트 반환


# from langchain.embeddings.base import Embeddings

# class TitanEmbeddings(Embeddings):
#     def embed_documents(self, texts):
#         return [titan_embed(t) for t in texts]

#     def embed_query(self, text):
#         return titan_embed(text)
    
# from langchain.embeddings.base import Embeddings

# class TitanEmbeddings(Embeddings):
#     def embed_documents(self, texts):
#         return [titan_embed(t) for t in texts]

#     def embed_query(self, text):
#         return titan_embed(text)
    
# def insert_chromadb():
#     try:
#         client.delete_collection("prompt_collection")
#     except Exception:
#         pass  # 없으면 무시

#     #  Embeddings 클래스 객체 사용
#     titan_embeddings = TitanEmbeddings()

#     vectorstore = Chroma(
#         client=client,
#         collection_name="prompt_collection",
#         embedding_function=titan_embeddings  
#     )
    
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     vectorstore.add_texts(
#         texts=[p["sample_prompt"] for p in prompts],
#         metadatas=[{
#             "intent": p["intent"],
#             "description": p["description"],
#             "response_type": p["response_type"],
#             "examples": ", ".join(p["examples"])
#         } for p in prompts],
#         ids=[f"doc_{i}" for i in range(len(prompts))]
#     )
    
#     #  테스트 검색
#     query = "자기소개 해줘"
#     results = vectorstore.similarity_search(query, k=1)
#     print("검색 결과:", results[0].page_content, flush=True)  

