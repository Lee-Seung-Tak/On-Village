from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from pydub import AudioSegment
from .util.util import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
from typing import Tuple
import asyncio
import wave
import os


SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16bit

async def pcm_to_wav(pcm_bytes: bytes, sample_width=SAMPLE_WIDTH, sample_rate=SAMPLE_RATE, filename="user_audio.wav") -> str:
    os.makedirs("wav", exist_ok=True)
    filepath = os.path.join("wav", filename)
    
    # 파일명 자동 증가
    count = 1
    name, ext = os.path.splitext(filename)
    while os.path.exists(filepath):
        filepath = os.path.join("wav", f"{name}_copy{count}{ext}")
        count += 1

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)            # mono
        wf.setsampwidth(sample_width) # 16bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    return filepath



# class StringEventHandler(TranscriptResultStreamHandler):
#     """PCM 스트리밍 결과를 문자열로 수집"""
#     def __init__(self, output_stream):
#         super().__init__(output_stream)
#         self.transcript_parts = []

#     async def handle_transcript_event(self, transcript_event: TranscriptEvent):
#         for result in transcript_event.transcript.results:
#             for alt in result.alternatives:
#                 self.transcript_parts.append(alt.transcript)

#     def get_transcript(self):
#         return " ".join(self.transcript_parts).strip()

# async def transcribe_wav_to_text(
#     wav_path: str,
#     language_code: str = "ko-KR",
#     chunk_size: int = 1024 * 8
# ) -> str:
#     """
#     WAV 파일을 읽어서 AWS Transcribe Streaming API로 전송 후 텍스트로 변환합니다.

#     :param wav_path: 변환할 WAV 파일 경로
#     :param language_code: 언어 코드 (기본값: 한국어)
#     :param chunk_size: 전송할 오디오 청크 크기 (바이트 단위)
#     :return: 인식된 텍스트 문자열
#     """
    
#     # 1. 파일 존재 여부 확인
#     if not os.path.exists(wav_path):
#         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {wav_path}")

#     # 2. WAV 파일 열기 (PCM 데이터 추출)
#     with wave.open(wav_path, "rb") as wf:
#         sample_rate = wf.getframerate()
#         audio_bytes = wf.readframes(wf.getnframes())  # raw PCM 데이터

#     # 3. AWS Transcribe Streaming 클라이언트 생성
#     client = TranscribeStreamingClient(region="us-west-2")

#     # 4. 스트리밍 세션 시작
#     stream = await client.start_stream_transcription(
#         language_code=language_code,
#         media_encoding="pcm",
#         media_sample_rate_hz=sample_rate
#     )

#     handler = StringEventHandler(stream.output_stream)

#     # 5. PCM 데이터를 chunk_size 단위로 잘라서 전송
#     async def send_chunks():
#         for i in range(0, len(audio_bytes), chunk_size):
#             chunk = audio_bytes[i:i+chunk_size]
#             await stream.input_stream.send_audio_event(audio_chunk=chunk)
#         await stream.input_stream.end_stream()

#     # 6. 오디오 전송과 이벤트 수신을 동시에 실행
#     await asyncio.gather(send_chunks(), handler.handle_events())

#     # 7. 최종 인식된 텍스트 반환
#     print("handler.get_transcript() : ", handler.get_transcript() ,flush=True)
#     return handler.get_transcript()



class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.final_transcript = ""  # 최종 텍스트 누적

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if result.is_partial:
                continue  # 중간 결과 무시
            for alt in result.alternatives:
                self.final_transcript += alt.transcript + " "  # 누적

    def get_transcript(self):
        return self.final_transcript.strip()

async def transcribe_wav_to_text(file_path):
    # Set up our client with your chosen Region
    client = TranscribeStreamingClient(region="ap-northeast-2")

    # Start transcription to generate async stream
    stream = await client.start_stream_transcription(
        language_code="ko-KR",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    # WAV 파일을 읽고 스트리밍으로 전송
    async def send_wav_stream(stream, wav_path):
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        chunk_size = 1024 * 16
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i+chunk_size]
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        await stream.input_stream.end_stream()

    # Instantiate our handler
    handler = MyEventHandler(stream.output_stream)

    # WAV 전송과 이벤트 핸들러 동시에 실행
    await asyncio.gather(send_wav_stream(stream, file_path), handler.handle_events())
    return handler.get_transcript()
# import boto3
# import time
# import uuid
# transcribe = boto3.client("transcribe", region_name="ap-northeast-2")  # 서울 리전

# def transcribe_wav_file(wav_s3_url: str, language_code="ko-KR"):

#     job_name = f"transcribe_job_{uuid.uuid4()}"
    
#     transcribe.start_transcription_job(
#         TranscriptionJobName=job_name,
#         Media={"MediaFileUri": wav_s3_url},
#         MediaFormat="wav",
#         LanguageCode=language_code
#     )

#     # 완료 대기
#     while True:
#         status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
#         if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
#             break
#         time.sleep(1)

#     if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
#         result_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
#         import requests
#         transcript_json = requests.get(result_url).json()
#         return transcript_json["results"]["transcripts"][0]["transcript"]
#     else:
#         return None
# 사용 예시
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("사용법: python transcribe_local.py audio_file.wav")
#         sys.exit(1)

#     audio_file = sys.argv[1]
#     result = asyncio.run(transcribe_local_file(audio_file))
#     print("변환 결과:", result)
