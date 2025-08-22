
from langchain.prompts import ChatPromptTemplate
import boto3
import json
from .util.util import grandparent_agent, output_parser
from grandparent_agent.generate.retriever import retriever, prompt_template

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


async def rag_to_speech(user_input: str, voice_id="Seoyeon", engine="neural"):
    # 3-1) 리트리버로 관련 프롬프트 검색
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs]) if docs else "관련 자료 없음"
    print("context :" , context, flush=True)
    # 3-2) LLM 호출
    chain = prompt_template | grandparent_agent | output_parser
    response_text = chain.invoke({"question": user_input, "context": context})

    # 3-3) AWS Polly TTS
    polly = boto3.client("polly", region_name="ap-northeast-2")
    tts_response = polly.synthesize_speech(
        Text=response_text,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine=engine
    )
    audio_bytes = tts_response["AudioStream"].read()

    # 3-4) 텍스트 + 오디오 반환
    return {
        "text": response_text,
        "audio_bytes": audio_bytes
    }
    
async def text_to_speech_aws_polly(text, voice_id="Seoyeon", engine="neural"):
    polly = boto3.client("polly", region_name="ap-northeast-2")
    
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine=engine
    )

    audio_bytes = response["AudioStream"].read()  # 바이너리 데이터 추출

    # WebSocket 바이너리 전송
    return audio_bytes
