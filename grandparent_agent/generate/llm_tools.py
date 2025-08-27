

import requests
from dotenv import load_dotenv
import os

# 프로젝트 루트 기준 절대 경로
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
load_dotenv(dotenv_path)

# tools 정의
def web_search_tool(query: str) -> str:
    """
    웹 검색 도구 (Stub 예시)
    실제 구현에서는 Google Search API, Naver API 등으로 연동 가능
    """
    return f"[웹 검색 결과] '{query}' 에 대한 정보를 찾아왔습니다."

import json
def llm_weather_tool( location ) -> dict:
    """
    날씨 조회 도구
    OpenWeatherMap API를 활용하여 실시간 날씨 데이터 반환
    """
    
    
    OpenWeather_KEY='95eaeaa646dddbb23a63d1551244e97b'
    print("llm_weather_tool location : ", location , flush=True)
    location_data = json.loads(location)
    lat = location_data['lat']
    lon = location_data['lon']
    api_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={OpenWeather_KEY}&units=metric"
    )
    print("api_url : ", api_url , flush=True)
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return json.dumps(response.json())  # JSON 데이터 반환
    else:
        return {"error": f"API 요청 실패 (status: {response.status_code})"}


def emergency_call_tool() -> str:
    """
    긴급 상황 대응 도구
    실제 구현에서는 119 연결 API, 혹은 내부 알림 시스템 연동 가능
    """
    return "긴급 상황이 감지되어 119에 신고 절차를 시작합니다."
