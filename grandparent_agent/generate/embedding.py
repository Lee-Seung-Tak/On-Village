import boto3, json
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from .prompts import prompts
# Bedrock runtime client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-embed-text-v2:0"

import chromadb
from langchain.vectorstores import Chroma

# ChromaDB 클라이언트 연결
client = chromadb.HttpClient(host="chroma", port=8000)
# 'prompt_collection' 컬렉션 생성 또는 가져오기
collection = client.get_or_create_collection(name="prompt_collection")



def titan_embed(text: str):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(modelId=model_id, body=body)
    model_response = json.loads(response["body"].read())
    return model_response["embedding"]  # 1024차원 리스트 반환


class TitanEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [titan_embed(t) for t in texts]

    def embed_query(self, text):
        return titan_embed(text)
    
def insert_chromadb():
    try:
        client.delete_collection("prompt_collection")
    except Exception:
        pass  # 없으면 무시

    #  Embeddings 클래스 객체 사용
    titan_embeddings = TitanEmbeddings()

    vectorstore = Chroma(
        client=client,
        collection_name="prompt_collection",
        embedding_function=titan_embeddings  
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    vectorstore.add_texts(
        texts=[p["sample_prompt"] for p in prompts],
        metadatas=[{
            "intent": p["intent"],
            "description": p["description"],
            "response_type": p["response_type"],
            "examples": ", ".join(p["examples"])
        } for p in prompts],
        ids=[f"doc_{i}" for i in range(len(prompts))]
    )
    
    #  테스트 검색
    query = "자기소개 해줘"
    results = vectorstore.similarity_search(query, k=1)
    print("검색 결과:", results[0].page_content, flush=True)  

