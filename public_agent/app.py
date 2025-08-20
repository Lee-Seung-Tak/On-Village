# app_streamlit_video_agent.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve()).split('/public_agent/')[0])
import os, json, inspect
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from public_agent.prompts.prompt import AGENT_SYSTEM_PROMPT
from public_agent.tools.video_tool import generate_video
from public_agent.tools.prompt_tool import fill_video_fields_tool
from public_agent.tools.festival_idea_tool import festival_idea_tool

load_dotenv()
st.set_page_config(page_title="Video Agent (GPT-4.1-nano → Veo tool)", page_icon="🎬")
st.title(" GPT-4.1-nano로 브리프 받고, Veo 툴로 영상 생성")

# -------- Tools schema (LangChain) --------
class VideoSpec(BaseModel):
    subject: str = Field(..., description="영상의 주 피사체/인물/대상")
    context: str = Field(..., description="배경/장소/상황")
    action: str = Field(..., description="핵심 동작/대사(있다면)")
    style: str = Field(..., description="시각 스타일/레퍼런스")
    camera_motion: str = Field(..., description="카메라 무브/샷 유형")
    composition: str = Field(..., description="구도/렌즈/프레이밍")
    ambiance: str = Field(..., description="분위기/조명/사운드 텍스처")
    negative_prompt: str = Field(..., description="피하고 싶은 요소")

video_tool = StructuredTool.from_function(
    func=generate_video,
    name="generate_video",
    description="모든 필드로 Veo(16:9, 8초) 생성. JSON(status/gcs_uri/local_file/prompt) 반환.",
    args_schema=VideoSpec,
)

# -------- LLM + Agent --------
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
SYSTEM = AGENT_SYSTEM_PROMPT + r"""
당신은 '대화형 멀티툴 에이전트'다.

[대화 라우터(최상위 분기)]
- 새 대화거나 모호하면 먼저 묻는다: "지역축제 영상 제작을 원하시나요, 아니면 일반 대화/다른 업무를 원하시나요?"
- '일반'이면 도구 호출 없이 자연스럽게 답한다.
- '지역축제 영상'이면 묻는다: "A) 최근 어르신 대화에서 2가지 아이디어 추천" 또는 "B) 직접 원하는 내용으로 제작"

[분기 A]
1) festival_idea_tool(path="public_agent/conversation.json", k=2) 호출
2) 응답은 아이디어 JSON만 출력
3) 사용자가 1개 고르면 그 seed로 8필드 채움
4) 8필드가 채워지면 fill_video_fields_tool 호출해 정규화
5) 결과는 8필드 JSON만 출력 (subject, context, action, style, camera_motion, composition, ambiance, negative_prompt)

[분기 B]
1) 대화로 8필드 채움
2) 8필드가 채워지면 fill_video_fields_tool 호출해 정규화
3) 결과는 8필드 JSON만 출력

[스타일 규칙]
- 기본: cartoon/animation, human characters
- negative_prompt에는 photorealistic, live-action, hyper-realistic, uncanny 포함

[주의]
- generate_video는 여기서 호출하지 않는다(생성은 UI에서 처리)
- JSON 반환 시 JSON만 출력(여분 텍스트 금지)
- 툴 실패 시 이유만 말하고 다음 단계 유도
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
tools = [fill_video_fields_tool, video_tool, festival_idea_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# -------- State --------
ss = st.session_state
for k, v in {
    "chat_history": [],
    "pending_fields": None,
    "awaiting_confirm": False,
    "awaiting_edit_field": None,
    "video_branch_mode": "ask",
    "awaiting_idea_choice": False,
    "idea_options": None,
}.items():
    if k not in ss: ss[k] = v

# -------- Helpers --------
def _coerce_to_dict(raw):
    return json.loads(raw) if isinstance(raw, str) else raw

def call_fill_fields(seed: str) -> dict:
    fn = fill_video_fields_tool.func

    # ---- 1) 함수 시그니처 직접 호출 (가장 우선) ----
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        # 8필드 직접 받는 형태
        eight = {"subject","context","action","style","camera_motion","composition","ambiance","negative_prompt"}
        if set(params) >= eight:
            raw = fn(
                subject=seed,
                context="",
                action="",
                style="cartoon / animation, human characters",
                camera_motion="",
                composition="",
                ambiance="",
                negative_prompt="photorealistic, live-action, hyper-realistic, uncanny",
            )
            return _coerce_to_dict(raw)

        # user_input + keywords 동시 요구
        if {"user_input","keywords"}.issubset(params):
            return _coerce_to_dict(fn(user_input=seed, keywords=seed))

        # 단일/부분 조합
        if "user_input" in params:
            return _coerce_to_dict(fn(user_input=seed))
        if "keywords" in params:
            return _coerce_to_dict(fn(keywords=seed))
        if len(params) == 1:
            return _coerce_to_dict(fn(**{params[0]: seed}))
    except Exception:
        pass

    # ---- 2) LangChain Tool args_schema 기반 호출 ----
    try:
        schema = getattr(fill_video_fields_tool, "args_schema", None)
        field_names = []
        if schema is not None:
            if hasattr(schema, "model_fields"):
                field_names = list(schema.model_fields.keys())     # pydantic v2
            elif hasattr(schema, "__fields__"):
                field_names = list(schema.__fields__.keys())       # pydantic v1

        if {"user_input","keywords"}.issubset(field_names):
            raw = fill_video_fields_tool.invoke({"user_input": seed, "keywords": seed})
            return _coerce_to_dict(raw)

        for k in ["user_input", "keywords"]:
            if k in field_names:
                raw = fill_video_fields_tool.invoke({k: seed})
                return _coerce_to_dict(raw)

        if field_names:
            raw = fill_video_fields_tool.invoke({field_names[0]: seed})
            return _coerce_to_dict(raw)
    except Exception:
        pass

    # ---- 3) 가능한 조합 전부 시도 ----
    try:
        raw = fill_video_fields_tool.invoke({"user_input": seed, "keywords": seed})
        return _coerce_to_dict(raw)
    except Exception:
        pass
    for k in ["user_input","keywords","seed","input","prompt","query","topic","brief","idea","text","spec"]:
        try:
            raw = fill_video_fields_tool.invoke({k: seed})
            return _coerce_to_dict(raw)
        except Exception:
            continue

    try:
        return _coerce_to_dict(fn(user_input=seed, keywords=seed))
    except Exception:
        pass
    for k in ["user_input","keywords","seed","input","prompt","query","topic","brief","idea","text","spec"]:
        try:
            return _coerce_to_dict(fn(**{k: seed}))
        except Exception:
            continue

    raise RuntimeError("fill_video_fields_tool 호출 실패: user_input/keywords 동시 및 모든 폴백 거부")

def apply_cartoon_defaults(fields: dict) -> dict:
    if not isinstance(fields, dict): return fields
    style = str(fields.get("style", ""))
    if ("cartoon" not in style.lower()) and ("animation" not in style.lower()):
        fields["style"] = (style + ", cartoon / animation, human characters").strip(", ")
    neg = str(fields.get("negative_prompt", ""))
    add_neg = "photorealistic, live-action, hyper-realistic, uncanny"
    if all(k not in neg.lower() for k in ["photorealistic","live-action","hyper-realistic","uncanny"]):
        fields["negative_prompt"] = (neg + ", " + add_neg).strip(", ")
    return fields

def _is_json_text(s: str) -> bool:
    try: json.loads(s); return True
    except Exception: return False

def _is_fields_json(d: dict) -> bool:
    need = {"subject","context","action","style","camera_motion","composition","ambiance","negative_prompt"}
    return isinstance(d, dict) and need.issubset(d.keys()) and "status" not in d

def _is_ideas_json(d):
    if isinstance(d, dict) and isinstance(d.get("ideas"), list): return True
    if isinstance(d, list) and d and isinstance(d[0], dict) and (("title" in d[0]) or ("seed" in d[0]) or ("summary" in d[0])): return True
    return False

FIELD_UI = [
    ("subject","🧑‍🎤 주요 피사체/인물"),("context","📍 배경/장소/상황"),("action","🎬 핵심 동작/대사"),
    ("style","🎨 시각 스타일"),("camera_motion","🎥 카메라 무브/샷"),("composition","🖼️ 구도/렌즈/프레이밍"),
    ("ambiance","🔊 분위기/조명/사운드"),("negative_prompt","🚫 피하고 싶은 요소"),
]
def _pretty_fields(fields: dict):
    st.markdown("### 📝 브리프 요약")
    for k, label in FIELD_UI:
        st.markdown(f"- **{label}**: {str(fields.get(k,'—')) or '—'}")
    st.markdown("확정하려면 **예/생성/확정**, 수정은 **수정 {필드명}** (예: `수정 대사`).")

def _render_ideas(d):
    ideas = d.get("ideas") if isinstance(d, dict) else d
    st.markdown("### 🎯 추천 아이디어")
    for i, it in enumerate(ideas, 1):
        title = it.get("title") or it.get("seed") or it.get("summary") or "아이디어"
        reason = it.get("reason") or it.get("rationale") or ""
        seed = it.get("seed") or ""
        st.markdown(f"**{i}. {title}**")
        if reason: st.markdown(f"- 근거: {reason}")
        if seed and seed != title: st.markdown(f"- 시드: {seed}")
    st.markdown("➡️ 하나를 선택(번호/제목)해 주세요.")

# -------- Welcome once --------
if len(ss.chat_history) == 0:
    ss.chat_history.append(AIMessage(content="지역축제 **영상 제작**을 원하시나요, 아니면 **일반 대화/다른 업무**를 원하시나요? (예: '영상', '일반')"))

# -------- Render history (hide raw JSON) --------
for m in ss.chat_history:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        content = m.content
        if _is_json_text(content):
            try:
                obj = json.loads(content)
                if _is_fields_json(obj): _pretty_fields(obj); continue
                if _is_ideas_json(obj): _render_ideas(obj); continue
                if isinstance(obj, dict) and obj.get("status"): st.markdown("🎥 **이전 생성 결과가 있습니다.**"); continue
            except Exception: pass
        st.markdown(content)

# -------- Main loop --------
user_input = st.chat_input("메시지를 입력하세요.")
if not user_input: st.stop()
norm = user_input.strip().lower().strip(" .!?,")
yes_tokens = {"예","네","넵","네네","생성","확정","ok","okay","go","yes","y"}

# 0) 필드 수정값 입력 단계
if ss.awaiting_edit_field and ss.pending_fields:
    ss.pending_fields[ss.awaiting_edit_field] = user_input
    ss.awaiting_edit_field = None
    ss.awaiting_confirm = True
    ss.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"): _pretty_fields(ss.pending_fields)
    st.stop()

# 1) 8필드 확정 단계
if ss.awaiting_confirm and ss.pending_fields:
    ss.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.markdown(user_input)
    if norm in yes_tokens:
        with st.chat_message("assistant"):
            try:
                with st.spinner("영상 생성 중..."):
                    video_result_text = generate_video(**ss.pending_fields)
                v = json.loads(video_result_text)
                if isinstance(v, dict) and v.get("status") == "success":
                    st.success("✅ 비디오가 생성되었습니다! 확인해보세요!")
                    # 🔽 경로는 노출하지 않고, 미리보기 + 다운로드만 제공
                    local_path = v.get("local_file")
                    if local_path and os.path.exists(local_path):
                        st.video(local_path)
                        with open(local_path, "rb") as f:
                            st.download_button("📥 MP4 다운로드", f, file_name=os.path.basename(local_path))
                    else:
                        st.warning("로컬 파일을 찾지 못했습니다. 잠시 후 다시 시도해주세요.")
                else:
                    st.error("영상 생성 실패")
            except Exception as e:
                st.error(f"generate_video 실패: {e}")
        ss.awaiting_confirm = False
        ss.pending_fields = None
        ss.video_branch_mode = "ask"
        ss.awaiting_idea_choice = False
        ss.idea_options = None
        st.stop()
    else:
        if norm.startswith("수정"):
            key_map = {
                "subject":["subject","주제","피사체","인물"],
                "context":["context","배경","장소","상황"],
                "action":["action","대사","행동","동작"],
                "style":["style","스타일","레퍼런스","룩"],
                "camera_motion":["camera_motion","카메라","카메라무브","무브","샷"],
                "composition":["composition","구도","렌즈","프레이밍"],
                "ambiance":["ambiance","분위기","조명","사운드","사운드텍스처"],
                "negative_prompt":["negative_prompt","네거티브","제외","피하고"],
            }
            t = norm.replace(" ",""); target=None
            for k, aliases in key_map.items():
                if any(a.replace(" ","") in t for a in aliases): target=k; break
            if target:
                ss.awaiting_edit_field = target
                with st.chat_message("assistant"):
                    st.markdown(f"✏️ **{target}** 필드 새 값을 보내주세요.")
                st.stop()
        with st.chat_message("assistant"):
            st.markdown("확정은 **예/생성/확정**, 수정은 **수정 {필드명}** (예: `수정 대사`).")
        st.stop()

# 2) 최상위 분기
if ss.video_branch_mode == "ask":
    ss.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.markdown(user_input)
    if any(k in norm for k in ["영상","video","동영상","비디오","축제"]):
        ss.video_branch_mode = "video_choice"
        with st.chat_message("assistant"):
            st.markdown("진행 방식을 선택해주세요:\n\n**A)** 어르신 대화 기반 **아이디어 추천**\n**B)** **직접 원하는 영상** 제작\n\n→ **'A' / 'B'** 또는 **'추천' / '직접'**")
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("생각 중..."):
                    result = agent_executor.invoke({"input": user_input,"chat_history": ss.chat_history})
            except Exception as e:
                st.error(f"Agent 실패: {e}")
            else:
                out = result.get("output","")
                ss.chat_history.append(AIMessage(content=out))
                if _is_json_text(out):
                    try:
                        obj=json.loads(out)
                        if _is_fields_json(obj): _pretty_fields(obj)
                        elif _is_ideas_json(obj): _render_ideas(obj)
                        else: st.markdown(out)
                    except Exception: st.markdown(out)
                else:
                    st.markdown(out)
    st.stop()

# 3) A/B 선택
if ss.video_branch_mode == "video_choice":
    ss.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.markdown(user_input)
    choice = "A" if norm in {"a","추천","아이디어","idea","ideas"} else ("B" if norm in {"b","직접","direct"} else None)
    if choice == "A":
        ss.video_branch_mode = "A"
        with st.chat_message("assistant"):
            try:
                with st.spinner("어르신 대화에서 아이디어 추출 중..."):
                    raw = festival_idea_tool.func(path="public_agent/conversation.json", k=2)
                data = json.loads(raw)
                ss.idea_options = data.get("ideas") if isinstance(data, dict) else data
                ss.awaiting_idea_choice = True
                _render_ideas(data)
            except Exception:
                st.warning("아이디어 추천에 실패했어요. **직접 제작**으로 진행합니다. 키워드를 보내주세요.")
                ss.video_branch_mode = "B"
        st.stop()
    elif choice == "B":
        ss.video_branch_mode = "B"
        with st.chat_message("assistant"):
            st.markdown("좋아요. **직접 제작**으로 진행합니다. 원하는 **키워드/주제**를 한 줄로 보내주세요.")
        st.stop()
    else:
        with st.chat_message("assistant"): st.markdown("**A(추천)** 또는 **B(직접)** 중 선택해주세요.")
        st.stop()

# 4) 추천 아이디어 선택 → 8필드 생성
if ss.video_branch_mode == "A" and ss.awaiting_idea_choice:
    ss.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.markdown(user_input)
    picked = None; ideas = ss.idea_options or []
    try:
        idx=int(user_input.strip()); 
        if 1<=idx<=len(ideas): picked=ideas[idx-1]
    except Exception: pass
    if not picked:
        low=user_input.lower()
        for it in ideas:
            title=(it.get("title") or it.get("seed") or it.get("summary") or "")
            if title and title.lower() in low: picked=it; break
    if not picked:
        with st.chat_message("assistant"): st.markdown("번호(1/2)나 제목 일부로 다시 선택해주세요.")
        st.stop()
    seed = picked.get("seed") or picked.get("title") or picked.get("summary") or ""
    with st.chat_message("assistant"):
        try:
            with st.spinner("아이디어 기반 브리프 생성 중..."):
                fields = call_fill_fields(seed)
                fields = apply_cartoon_defaults(fields)
            ss.pending_fields = fields
            ss.awaiting_confirm = True
            ss.awaiting_idea_choice = False
            _pretty_fields(fields)
        except Exception as e:
            st.error(f"브리프 자동 생성 실패: {e}")
            st.markdown("원하시면 **직접 키워드**로 진행할 수 있어요. 예: `보령 머드축제 숏폼`")
    st.stop()

# 5) 직접 제작: 첫 seed → 8필드 생성
if ss.video_branch_mode == "B" and (ss.pending_fields is None):
    ss.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        try:
            with st.spinner("입력하신 주제로 브리프 생성 중..."):
                fields = call_fill_fields(user_input)
                fields = apply_cartoon_defaults(fields)
            ss.pending_fields = fields
            ss.awaiting_confirm = True
            _pretty_fields(fields)
        except Exception as e:
            st.error(f"브리프 자동 생성 실패: {e}")
    st.stop()

# 6) 일반 대화
ss.chat_history.append(HumanMessage(content=user_input))
with st.chat_message("user"): st.markdown(user_input)
with st.chat_message("assistant"):
    try:
        with st.spinner("생각 중..."):
            result = agent_executor.invoke({"input": user_input,"chat_history": ss.chat_history})
    except Exception as e:
        st.error(f"Agent 실패: {e}")
        result = {"output": ""}
    out = result.get("output","")
    ss.chat_history.append(AIMessage(content=out))
    if _is_json_text(out):
        try:
            obj=json.loads(out)
            if _is_ideas_json(obj):
                ss.video_branch_mode="A"; ss.awaiting_idea_choice=True
                ss.idea_options = obj.get("ideas") if isinstance(obj, dict) else obj
                _render_ideas(obj)
            elif _is_fields_json(obj):
                ss.pending_fields=obj; ss.awaiting_confirm=True; _pretty_fields(obj)
            else:
                st.markdown(out)
        except Exception:
            st.markdown(out)
    else:
        st.markdown(out)
