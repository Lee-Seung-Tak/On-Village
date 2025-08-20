# from langchain.prompts import ChatPromptTemplate
# from langchain_aws import ChatBedrock
# from langchain_core.output_parsers import StrOutputParser
# from .util.util import grandparent_agent
# def llm_generate() :
#     # 1. 모델 초기화 (Bedrock Claude)
   

#     # 2. 프롬프트 템플릿 정의
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "너는 친절한 자기소개 봇이다."),
#         ("user", "{input}")
#     ])

#     # 3. 출력 파서 (텍스트만 추출)
#     parser = StrOutputParser()

#     # 4. 체인 구성 (프롬프트 → LLM → 파서)
#     chain = prompt | grandparent_agent | parser

#     # 5. 실행
#     output_text = chain.invoke({"input": "자기소개해줘"})
#     print("모델 응답:", output_text)

#     # 6. TTS 처리 (예: AWS Polly)
#     import boto3
#     polly = boto3.client("polly")
#     tts = polly.synthesize_speech(Text=output_text, VoiceId="Seoyeon", OutputFormat="mp3")

#     with open("output.mp3", "wb") as f:
#         f.write(tts["AudioStream"].read())

# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json

from botocore.exceptions import ClientError
def llm_generate(): 
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Claude 3 Haiku.
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    # Define the prompt for the model.
    prompt = "안녕하세요 자기소개 해주세요."

    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)
    response = ''
    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)
        print("here -1 ",response, flush=True)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        # exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    print(response_text)


