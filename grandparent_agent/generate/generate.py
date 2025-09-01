
from langchain.prompts import ChatPromptTemplate
import boto3
import json
from .util.util import grandparent_agent, output_parser
from .retriever import init_rag_system
from .util.util import agent
def llm_generate( user_input: str ): 
    try:
        if not user_input :
            user_input = "안녕하세요, 다시 한번 말씀해주시겠어요?"
            
        messages = [
            ("system", f"당신은 AI assistant '기쁨이' 입니다. 당신의 말상대는 어르신들 입니다. 항상 예의를 갖추어 존댓말로 대답하세요."),
            ("user", "{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | grandparent_agent | output_parser

        response = chain.invoke({ "question": user_input })

        print("Response from model: ", response, flush=True)
        return response
    except Exception as e:
        print(f"ERROR: Can't invoke ==> Reason: {e}")


from .tts import text_to_speech_aws_polly

async def rag_to_speech(user_input: str, user_information, voice_id="Seoyeon", engine="neural"):
    # RAG 시스템 초기화
    vectorstore, prompt_template, top_k, score_threshold = init_rag_system()

    # similarity_search_with_score로 문서 필터링
    results = vectorstore.similarity_search_with_score(user_input, k=top_k)
    docs = [doc for doc, score in results if score >= score_threshold]
    input_text = ''
    if docs:
        context = "\n\n".join([d.page_content for d in docs])
        print("context : ", context, flush=True)
        input_text = f"prompt : {context}, user_input : {user_input}, user_information :{user_information}"
    else:
        context = None
        input_text = f"{user_input} 사용자 정보 {user_information}"

    # Agent 실행
    response_text = agent.run(input_text)

    # TTS 변환
    audio_bytes = await text_to_speech_aws_polly(response_text)

    return {
        "text": response_text,
        "audio_bytes": audio_bytes
    }