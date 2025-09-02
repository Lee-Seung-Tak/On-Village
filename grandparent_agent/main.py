from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from wake_word_model.wake_word import wwd_is_detected
from wake_word_model.wake_word import RECORD_BYTES, SLIDING_STEP_BYTES
from generate.stt import pcm_to_wav, transcribe_wav_to_text
from generate.util.util import SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS, CHUNK_DURATION_MS, MAX_SILENCE_COUNT, frame_size
from generate.util.util import vad, get_user_location
from generate.generate import rag_to_speech
import os
import base64
import json
status = {
    "idle": {
        "code": 0,
        "description": "대기 중"
    },
    "listening": {
        "code": 1,
        "description": "사용자의 음성을 듣는 중"
    },
    "thinking": {
        "code": 2,
        "description": "모델이 응답을 생성하는 중"
    },
    "speaking": {
        "code": 3,
        "description": "응답을 음성으로 출력하는 중"
    }
}

user_information = {
    "age" : "71",
    "name" : "도봉순",
    "disease" : "당뇨",
}
# Python dict → JSON 문자열로 변환 가능
# status_json = json.dumps(status, ensure_ascii=False, indent=4)
app = FastAPI()

# Mount static directory for web frontend
app.mount("/web", StaticFiles(directory="wwd_web", html=True), name="web")

# CORS middleware for development
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    print("[LOG] - connected cliend")
    
    await ws.accept()
    buffer            = bytearray()
    user_input_buffer = bytearray()
    
    try:
        while True:
            message = await ws.receive()
            # 1) 텍스트(JSON) 메시지 처리
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue
                    
                if payload.get("event") == "location":
                    user_information['location'] = get_user_location( payload )
                    
                
                    
                if payload.get("event") == "press_to_push":
                    # 눌러서 말하기 시작
                    await ws.send_text(json.dumps(status['listening']))
                    print("[Push-to-Talk] 음성 입력 시작", flush=True)

                    user_input_buffer = bytearray()
                    silence_count     = 0
                    stop_sig          = 0

                    # 2) 음성 데이터 수집 루프
                    while True:
                        audio_msg = await ws.receive()
                        
                        if "bytes" in audio_msg:
                            data = audio_msg["bytes"]
                            user_input_buffer.extend(data)

                            if len(user_input_buffer) >= frame_size:
                                frame = user_input_buffer[-frame_size:]
                                if not vad.is_speech(bytes(frame), SAMPLE_RATE):
                                    silence_count += 1
                                else:
                                    silence_count = 0
                                    stop_sig += 1

                            if silence_count >= MAX_SILENCE_COUNT or stop_sig > MAX_SILENCE_COUNT:
                                break

                    # 3) 수집된 음성 처리
                    await ws.send_text(json.dumps(status['thinking']))
                    user_audio_path = await pcm_to_wav(user_input_buffer, SAMPLE_WIDTH, SAMPLE_RATE)
                    stt_data        = await transcribe_wav_to_text(user_audio_path)

                    print("user_audio_path:", user_audio_path, flush=True)
                    print("stt_data:", stt_data, flush=True)

                    # LLM + TTS
                    llm_response = await rag_to_speech(stt_data, json.dumps(user_information, ensure_ascii=False))

                    speaking_status = status["speaking"].copy()
                    speaking_status["audio_base64"] = base64.b64encode(llm_response['audio_bytes']).decode("utf-8")

                    await ws.send_text(json.dumps(speaking_status))

                    user_input_buffer = bytearray()
                    buffer            = bytearray()
            
            elif "bytes" in message:
           
                data = message["bytes"]
                buffer.extend(data)

                while len(buffer) >= RECORD_BYTES:
                    
                    window_bytes         = buffer[:RECORD_BYTES]
                    buffer               = buffer[SLIDING_STEP_BYTES:]
                    detected, confidence = wwd_is_detected(window_bytes)

                    if detected:
                        await ws.send_text(json.dumps(status['listening']))
                        print(f"[Detected] Talk ===> confidence_{confidence} ",flush=True)
                        silence_count = 0
                    
                        # 감지가 된 이후 사용자 음성 데이터 bytearr에 extend
                        stop_sig = 0
                        
                        while True:
                        
                            data = await ws.receive_bytes()
                
                            user_input_buffer.extend(data)
                        
                            # 최신 frame만 VAD 체크
                    
                            if len( user_input_buffer ) >= frame_size:
                                
                                frame = user_input_buffer[ -frame_size: ]
                                
                                if not vad.is_speech( bytes( frame ), SAMPLE_RATE ):
                                    silence_count += 1
                                    
                                else:
                                    silence_count = 0
                                    stop_sig      += 1
                        
                            if silence_count >= MAX_SILENCE_COUNT:
                                break
                            
                            if stop_sig > MAX_SILENCE_COUNT :
                                break
            
                            
                        # 전체 데이터를 그대로 WAV로 저장
                    
                        await ws.send_text(json.dumps(status['thinking']))
                        user_audio_path   = await pcm_to_wav(user_input_buffer, SAMPLE_WIDTH, SAMPLE_RATE)
                        stt_data          = await transcribe_wav_to_text(user_audio_path)
                        
                        print("user_audio_path : ", user_audio_path ,flush=True)
                        print("stt_data : ", stt_data ,flush=True)
                        
                        llm_resposne = await rag_to_speech(stt_data, json.dumps(user_information, ensure_ascii=False))

                    
                        
                        speaking_status                 = status["speaking"].copy()
                        speaking_status["audio_base64"] = base64.b64encode(llm_resposne['audio_bytes']).decode("utf-8")
                        await ws.send_bytes(json.dumps(speaking_status))
                    
                        
                        
                        buffer            = bytearray()
                        user_input_buffer = bytearray()
                    else:
                        print(f"웨이크워드 미감지 (신뢰도: {confidence:.4f})", flush=True)
                        await ws.send_text(f"웨이크워드 미감지 (신뢰도: {confidence:.4f})")                # WWD 감지가 된 경우

                
    except WebSocketDisconnect:
        print("클라이언트 연결 종료",flush=True)
    except Exception as e:
        print(f"웹소켓 오류: {e}", flush=True)
    finally:
        await ws.close()


@app.get("/conversation", response_class=HTMLResponse)
async def get():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_dir, 'wwd_web', 'index.html')
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=4000)