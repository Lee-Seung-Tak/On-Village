import chromadb
from langchain.vectorstores import Chroma

# ChromaDB 클라이언트 연결
client = chromadb.HttpClient(host="chroma", port=8000)
# 'prompt_collection' 컬렉션 생성 또는 가져오기
collection = client.get_or_create_collection(name="prompt_collection")

import boto3
import numpy as np
import json
from langchain.embeddings.base import Embeddings
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

from langchain.embeddings.base import Embeddings

class AWSEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        
        # body는 StreamingBody → .read() 해서 bytes 변환
        result = json.loads(response["body"].read())
        
        # Titan은 "embedding" 키에 벡터가 들어있음
        return result["embedding"]

    
aws_embeddings = AWSEmbeddings()

from langchain.prompts import ChatPromptTemplate

def init_rag_system( top_k=3 ):
    """
    1) Chroma 컬렉션 초기화
    2) retriever 생성
    3) ChatPromptTemplate 생성
    """
    # 컬렉션 재생성 (삭제 후)
    try:
        client.delete_collection("prompt_collection")
    except Exception:
        pass  # 컬렉션이 없으면 그냥 넘어감

    collection = client.get_or_create_collection(name="prompt_collection")

    # Vectorstore + Retriever
    vectorstore = Chroma(
        client=client,
        collection_name="prompt_collection",
        embedding_function=aws_embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # ChatPromptTemplate
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

    return retriever, prompt_template