
# access key 관련
from dotenv import load_dotenv
import os
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-northeast-2"  # 서울 리전




# VAD 관련
import webrtcvad

SAMPLE_RATE       = 16000
SAMPLE_WIDTH      = 2
CHANNELS          = 1
CHUNK_DURATION_MS = 30
MAX_SILENCE_COUNT = 17

vad = webrtcvad.Vad()
vad.set_mode(0)  # 0~3, 민감도
frame_size = int(SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000)


# llm 관련
import boto3
from langchain_aws import ChatBedrock

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,   # 반드시 본인 리전에 맞게 변경
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# LangChain용 Bedrock 래퍼
grandparent_agent = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0"
)