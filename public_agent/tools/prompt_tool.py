# public_agent/tools/prompt_tool.py
import json
import re
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

# 프롬프트 단계 모음 (step1=키워드추출, step2~7=필드 정제)
from public_agent.prompts import prompt as P


# ====== 최종 산출 스키마 ======
class VideoSpec(BaseModel):
    subject: str = Field(..., description="주 피사체/인물/대상")
    context: str = Field(..., description="배경/장소/상황")
    action: str = Field(..., description="핵심 동작/대사")
    style: str = Field(..., description="시각 스타일/레퍼런스")
    camera_motion: str = Field(..., description="카메라 무브/샷")
    composition: str = Field(..., description="구도/렌즈/프레이밍")
    ambiance: str = Field(..., description="분위기/조명/사운드")
    negative_prompt: str = Field(..., description="피하고 싶은 요소")


# ====== 유틸: 응답 → JSON(dict) 안전 파싱 ======
def _to_json_dict(text: str) -> Dict:
    print("[parse] raw:", str(text)[:400])
    t = str(text).strip()
    if t.startswith("```"):
        # ```json ... ``` 제거
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).strip()
        t = re.sub(r"\n?```$", "", t).strip()
    data = json.loads(t)
    if not isinstance(data, dict):
        raise ValueError("JSON은 객체여야 합니다.")
    return data


def _ensure_strings(d: Dict) -> Dict:
    keys = [
        "subject", "context", "action", "style",
        "camera_motion", "composition", "ambiance", "negative_prompt"
    ]
    return {k: str(d.get(k, "") or "") for k in keys}


# ====== 코어: 7단계 체인 ======
def run_7step_pipeline(
    user_input: str,
    keywords: List[str],                # seed 키워드 (반드시 유지)
    model: str = "gpt-4.1-nano",
    temperature: float = 0.2,
) -> Dict:
    """
    user_input + seed keywords를 받아 7단계 프롬프트 체이닝으로
    VideoSpec(JSON) dict을 반환. 각 단계는 print로 디버깅 가능.
    step1: 키워드 추출(JSON: {"keywords":[...], "user_input":"..."})
    step2~7: VideoSpec(JSON) 생성/정제
    """
    llm = ChatOpenAI(model=model, temperature=temperature)

    steps = [
        P.prompt_engineering1,  # 키워드 추출
        P.prompt_engineering2,  # 키워드 → 초기 필드 작성
        P.prompt_engineering3,  # 스타일 정제
        P.prompt_engineering4,  # 카메라/구도
        P.prompt_engineering5,  # 분위기/사운드/조명
        P.prompt_engineering6,  # 네거티브 프롬프트
        P.prompt_engineering7,  # 최종 검수
    ]

    current_json = None
    seed_copy = list(keywords)  # 로그용 원본 보관
    # 시드 유니크화(순서 보존)
    keywords = list(dict.fromkeys([str(k).strip() for k in (keywords or []) if str(k).strip()]))

    for i, make_prompt in enumerate(steps, start=1):
        print(f"\n[step{i}] === start ===")
        if i == 1:
            # step1은 '원문 시나리오 + seed 키워드'를 입력으로 보냄
            sys, usr = make_prompt(user_input, keywords)
        else:
            # step2~7은 "이전 JSON 문자열" + (병합된) 키워드 목록
            sys, usr = make_prompt(json.dumps(current_json, ensure_ascii=False), keywords)

        print(f"[step{i}] system:\n{sys}")
        print(f"[step{i}] user:\n{usr}")

        resp = llm.invoke([("system", sys), ("user", usr)])
        print(f"[step{i}] llm_resp:", str(resp.content)[:400])

        try:
            data = _to_json_dict(resp.content)
            # step1은 키워드 JSON이므로 _ensure_strings 적용 금지
            if i != 1:
                data = _ensure_strings(data)
        except Exception as e:
            print(f"[step{i}] JSON 파싱 실패:", e)
            raise

        # step1: 키워드 추출 → seed와 병합(순서: seed → extracted)
        if i == 1:
            current_json = data  # {"keywords":[...], "user_input":"..."}
            extracted = current_json.get("keywords", [])
            extracted = [str(k).strip() for k in (extracted or []) if str(k).strip()]
            keywords = list(dict.fromkeys(keywords + extracted))
            print(f"[step1] seed={seed_copy} / extracted={extracted} / merged={keywords}")
            print(f"[step1] === end ===")
            continue

        # step2~7: VideoSpec JSON 처리 → 키워드 포함 여부 점검
        dumped = json.dumps(data, ensure_ascii=False)
        missing = [k for k in keywords if k and k not in dumped]
        if missing:
            print(f"[step{i}] 현재 JSON에 누락 키워드:", missing)
        else:
            print(f"[step{i}] 모든 키워드 포함 OK")

        current_json = data
        print(f"[step{i}] === end ===")

    # 최종 검증 (최종 current_json은 VideoSpec여야 함)
    try:
        spec = VideoSpec(**current_json)
    except ValidationError as ve:
        print("[final] 검증 실패:", ve)
        raise
    print("[final] 검증 통과")
    return spec.dict()


# ====== LangChain Tool 래핑 ======
class FieldFillInput(BaseModel):
    user_input: str = Field(..., description="사용자 시나리오 또는 키워드 문장")
    keywords: List[str] = Field(..., description="반드시 유지해야 할 'seed' 키워드 리스트")


def field_fill_tool_fn(user_input: str, keywords: List[str]) -> str:
    """
    LangChain Tool 엔트리포인트.
    반환: 최종 VideoSpec JSON 문자열
    """
    result = run_7step_pipeline(user_input, keywords)
    return json.dumps(result, ensure_ascii=False)


fill_video_fields_tool = StructuredTool.from_function(
    func=field_fill_tool_fn,
    name="fill_video_fields",
    description="사용자 입력을 7단계로 정제하여 VideoSpec(JSON)으로 반환",
    args_schema=FieldFillInput,
)

