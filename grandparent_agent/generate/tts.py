import boto3
async def text_to_speech_aws_polly(text, voice_id="Seoyeon", engine="neural"):
    try:
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
    except Exception as e :
        print("tts function error : ", e ,flush=True)