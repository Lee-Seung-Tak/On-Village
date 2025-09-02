import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from public_agent.generate import llm_generate
import os

app = FastAPI()

# 1) API 엔드포인트 (/chat)
@app.get("/chat")
async def chat(user_input: str):
    try:
        response = llm_generate(user_input)
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 2) 정적 파일 마운트 (web/index.html → / 로 접근 가능)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # public_agent 경로
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "web"), html=True), name="static")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=True)
