from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from wake_word_model.wake_word import wwd_is_detected
from wake_word_model.wake_word import RECORD_BYTES, SLIDING_STEP_BYTES
import os

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
    await ws.accept()
    print("클라이언트 연결됨")
    buffer = bytearray()

    try:
        while True:
            data = await ws.receive_bytes()
            buffer.extend(data)

            # Process audio in sliding window
            while len(buffer) >= RECORD_BYTES:
                window_bytes = buffer[:RECORD_BYTES]
                buffer = buffer[SLIDING_STEP_BYTES:]

                detected, confidence = wwd_is_detected(window_bytes)
                if detected:
                    await ws.send_text(f"웨이크워드 감지됨! (신뢰도: {confidence:.4f})")
                    print(f"웨이크워드 감지됨! (신뢰도: {confidence:.4f})")
                    buffer = bytearray()  # Clear buffer after detection
                else:
                    await ws.send_text(f"웨이크워드 미감지 (신뢰도: {confidence:.4f})")
                    
    except WebSocketDisconnect:
        print("클라이언트 연결 종료")
    except Exception as e:
        print(f"웹소켓 오류: {e}")
    finally:
        await ws.close()

@app.get("/conversation/", response_class=HTMLResponse)
async def get():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_dir, 'wwd_web', 'index.html')
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=4000)