
from langchain.prompts import ChatPromptTemplate
import boto3
import json
from .util.util import grandparent_agent, output_parser
from .retriever import init_rag_system

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

async def rag_to_speech(user_input: str, voice_id="Seoyeon", engine="neural"):
    # 리트리버로 관련 프롬프트 검색
    retriever,prompt_template = init_rag_system()

    docs = retriever.invoke(user_input)
    context = "\n\n".join([d.page_content for d in docs]) if docs else "관련 자료 없음"
    print("context :" , context, flush=True)
    
    # LLM 호출
    chain = prompt_template | grandparent_agent | output_parser
    
    response_text = chain.invoke({"question": user_input, "context": context})
    print("response_text : ", response_text ,flush=True)
    audio_bytes   = await text_to_speech_aws_polly( response_text )


    # 텍스트 + 오디오 반환
    return {
        "text": response_text,
        "audio_bytes": audio_bytes
    }
    

