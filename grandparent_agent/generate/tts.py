import boto3
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-northeast-2"  # 서울 리전

def tts_korean_female(text: str, output_file: str = "output.mp3") -> None:
    """
        AWS Polly를 이용해 한국어 여성 음성 TTS 변환 후 로컬 파일 저장

        :param text: 변환할 텍스트
        :param output_file: 저장할 mp3 파일 경로
    """
    try:
        # Polly 클라이언트 생성
        polly = boto3.client(
            "polly",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )

        # TTS 요청
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat="mp3",       # 저장 형식
            VoiceId="Seoyeon",        # 한국어 여성 목소리
            LanguageCode="ko-KR"      # 한국어
        )

        # 파일 저장
        with open(output_file, "wb") as file:
            file.write(response["AudioStream"].read())

        print(f"TTS 변환 완료: {output_file}")

    except Exception as e:
        print(f"TTS 변환 중 오류 발생: {e}")


# 사용 예시
# if __name__ == "__main__":
#     sample_text = "안녕하세요. 오늘도 좋은 하루 되세요."
#     tts_korean_female(sample_text, "greeting.mp3")
