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
from .embedding import TitanEmbeddings

def init_rag_system(top_k=3, score_threshold=0.3):
    """
    1) Chroma 컬렉션 초기화
    2) vectorstore 생성
    3) ChatPromptTemplate 생성
    4) 직접 필터링 가능하게 반환
    """
    # Chroma 컬렉션 가져오기 (없으면 생성)
    collection = client.get_or_create_collection(name="prompt_collection")
    
    # Embeddings 객체 사용
    titan_embeddings = TitanEmbeddings()

    # Vectorstore
    vectorstore = Chroma(
        client=client,
        collection_name="prompt_collection",
        embedding_function=titan_embeddings
    )

    # ChatPromptTemplate
    system_template = """
        당신은 AI assistant '기쁨이'입니다. 어르신을 존중하며 존댓말로만 대답하세요.
        당신이 AI assistant고, 이러한 사실을 사용자가 모르게 하세요.
        확실하지 않은 답변은 모른다고 답변하세요.
        답변은 사용자의 질문파악 -> 질문이 정확한지 생각 -> 답변 순서로 생각하면서
        답변하세요.
        아래는 참고할 수 있는 프롬프트입니다:
        {context}
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{question}")
    ])

    # retriever 대신 vectorstore와 threshold 반환
    return vectorstore, prompt_template, top_k, score_threshold
