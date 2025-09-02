
# access key 관련
from dotenv import load_dotenv
import os
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "us-east-1"  # 서울 리전



# VAD 관련
import webrtcvad

SAMPLE_RATE       = 16000
SAMPLE_WIDTH      = 2
CHANNELS          = 1
CHUNK_DURATION_MS = 20
MAX_SILENCE_COUNT = 15

vad = webrtcvad.Vad()
vad.set_mode(0)  # 0~3, 민감도
frame_size = int(SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000)


# llm 관련
import boto3
from langchain_aws import ChatBedrock

aws_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"


# model_kwargs = {
#     "max_tokens": 5000,
#     "temperature": 0.5,
#     "top_k": 250,
#     "top_p": 1,
#     "stop_sequences": ["\n\nHuman"]
# }

# from langchain.tools import Tool
# from ..llm_tools import llm_weather_tool
# weather_tool = Tool(
#     name="weather_tool",
#     func=llm_weather_tool,  # 실제 날씨 조회 함수
#     description="사용자의 위도와 경도를 받아 날씨 정보를 반환"
# )

# tools = [weather_tool]
# # LangChain's Bedrock Wrapper (ChatBedrock)
# grandparent_agent = ChatBedrock(
#     region_name=AWS_REGION,
#     client=aws_client,
#     model_id=model_id,
#     model_kwargs=model_kwargs
# )

# from langchain_core.output_parsers import StrOutputParser
# output_parser = StrOutputParser()

# from langchain.agents import initialize_agent

# agent = initialize_agent(
#     tools=tools,
#     llm=grandparent_agent,  # ChatBedrock
#     agent="zero-shot-react-description",
#     verbose=True
# )

def get_user_location( payload ):
    lat = payload.get("lat")
    lon = payload.get("lon")
    return {"lat": lat, "lon": lon}