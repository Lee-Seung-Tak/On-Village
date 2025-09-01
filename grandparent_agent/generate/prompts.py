import chromadb
from langchain_chroma import Chroma   

# ChromaDB 클라이언트 연결
client = chromadb.HttpClient(host="chroma", port=8000)

# 'prompt_collection' 컬렉션 생성 또는 가져오기
collection = client.get_or_create_collection(name="prompt_collection")

# 데이터 정의
prompts = [
    {
        "intent": "자기소개",
        "description": "AI 자신을 소개하는 요청",
        "examples": ["기쁨이 서비스가 뭐야?","자기소개 해줘","넌 뭘 할 수 있어?","너의 기능이 뭐야"],
        "response_type": "system_info",
        "tool_name": None,  
        "sample_prompt": """저는 어르신들께 소소한 기쁨을 드리고자 만들어진 인공지능 에이전트 기쁨이에요. 저는 어르신들의 하루의 소중한 일상에 대해서
        이야기 나누는것을 좋아해요. 그리고 어르신들의 약 시간을 알려드릴 수 있어요. 그리고 재미있는 이야기도 들려드릴 수 있고, 어르신들께서 마을에서 
        생활하다 발견하는 불편사항을 저한테 말씀해주시면 정리해서 시에 전달할 수 있어요."""
    },
    
    {
        "intent": "정보 검색",
        "description": "어떤 정보에 대한 질문",
        "examples": ["김치찌개 끓이는 법","음식 레시피 알려줘","약국 전화번호 알려줘", "병원 몇시까지 하는지 알려줘", ],
        "response_type": "tool_call",
        "tool_name": "web_search_tool",  
        "sample_prompt": """사용자의 질문을 토대로 구글에 검색을 하고, 해당 결과를 바탕으로 답변합니다."""
    },
    
    {
        "intent": "날씨 질문",
        "description": "날씨 질문",
        "examples": ["오늘 날씨 어때","서울 날씨 어때" ,"의정부 날씨 어때"],
        "response_type": "tool_call",
        "tool_name": "weather_tool",  
        "sample_prompt": """제공된 날씨 데이터를 기반으로 날씨 정보를 답변합니다."""
    },
        
    {
        "intent": "긴급 신고",
        "description": "사용자의 응급 상황 판단 (응답 없음 3회 이상 포함)",
        "examples": ["도와줘", "살려줘", "응급 상황이야"],
        "response_type": "system_action",
        "tool_name": "emergency_call_tool",
        "sample_prompt": """긴급 상황이 감지되었습니다. 즉시 119에 신고 절차를 진행합니다."""
    },
    
]



