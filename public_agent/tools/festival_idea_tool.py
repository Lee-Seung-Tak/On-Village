# public_agent/tools/festival_idea_tool.py
# -*- coding: utf-8 -*-
import json, os, logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# --- logger ---
logger = logging.getLogger("festival_idea_tool")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
# 기본 레벨은 INFO (앱에서 DEBUG로 올릴 수 있음)
logger.setLevel(logging.INFO)

# ---------------- I/O ----------------
def load_conversation_json(path: str) -> List[dict]:
    logger.debug(f"load_conversation_json: path={path}")
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "turns" in data:
        data = [data]
    if not isinstance(data, list):
        logger.error("Unsupported JSON shape (expected list)")
        raise ValueError("Unsupported JSON shape (expected list of conversation blocks).")
    logger.info(f"Loaded conversations: {len(data)} blocks")
    return data

# -------------- 전처리 ---------------
def flatten_turns_to_text(blocks: List[dict], speaker: Optional[str] = "어르신") -> str:
    total_turns, kept = 0, 0
    lines: List[str] = []
    for conv in blocks:
        for t in conv.get("turns", []):
            total_turns += 1
            spk = (t.get("speaker") or "").strip()
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            if speaker and spk != speaker:
                continue
            kept += 1
            lines.append(f"{spk}: {txt}")
    logger.info(f"Turns total={total_turns}, kept={kept} (speaker={speaker})")
    preview = "\n".join(lines[:3])
    logger.debug(f"First lines preview:\n{preview}")
    return "\n".join(lines)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, max_chunks: int = 10) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    logger.info(f"Chunks created: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap})")
    for i, ch in enumerate(chunks[:3]):
        logger.debug(f"[chunk {i}] len={len(ch)} | preview={ch[:120].replace(os.linesep,' ')}")
    return chunks[:max_chunks]

# -------------- 모델 호출 --------------
def llm_extract_ideas(context_chunks: List[str], k: int, model: str = "gpt-4.1-nano", debug: bool = False) -> str:
    if not context_chunks:
        logger.error("Empty context: no chunks to send to LLM")
        return json.dumps({"status": "error", "message": "Empty context"}, ensure_ascii=False)

    context = "\n\n".join(context_chunks)
    logger.info(f"Sending to LLM: chunks={len(context_chunks)}, total_chars={len(context)} k={k} model={model}")

    sys_prompt = (
        "너는 로컬 축제 기획 보조 분석가다. 아래는 지역 어르신들의 실제 대화 발화다. "
        f"이 데이터를 근거로 지역 축제/관광 프로모션에 바로 적용 가능한 구체적 프로그램 아이디어를 정확히 {k}개 도출하라. "
        "한국 로컬 맥락, 어르신 언급 빈도, 접근성/편의, 실행 가능성을 고려하라. 반드시 JSON만 출력하라."
    )
    user_prompt = f"""[대화 컨텍스트]
{context}

[출력 스키마]
{{
  "status": "success",
  "ideas": [
    {{
      "title": "간결한 아이디어 명",
      "reason": "어르신 대화에서 이 아이디어가 유효한 근거 1문장",
      "keywords": ["핵심 키워드", "2~5개"],
      "seed": "후속 비디오 스펙 생성에 쓸 한 문장 시드(간결)",
      "evidence": ["짧은 근거 문장 1", "짧은 근거 문장 2"]
    }}
  ]
}}
JSON 외 텍스트 금지.
아이디어 개수는 정확히 {k}개."""

    try:
        llm = ChatOpenAI(model=model, temperature=0.2)
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        content = (resp.content or "").strip()
        logger.debug(f"Raw LLM content (truncated): {content[:1200]}")
        data = json.loads(content)
        if not isinstance(data, dict) or "ideas" not in data:
            logger.error("Invalid JSON structure from model")
            raise ValueError("Invalid JSON structure from model")
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.exception("LLM call/parse failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

# -------------- 툴 엔트리 --------------
class FestivalIdeaArgs(BaseModel):
    path: str = Field(default="public_agent/conversation.json", description="대화 JSON 파일 경로")
    k: int = Field(default=2, ge=1, le=5, description="추천 아이디어 개수")
    speaker: Optional[str] = Field(default="어르신", description="필터링 화자(None이면 전체)")
    chunk_size: int = Field(default=1200, description="청킹 문자 수")
    chunk_overlap: int = Field(default=200, description="청킹 오버랩")
    max_chunks: int = Field(default=10, description="LLM에 투입할 최대 청크 수")
    debug: bool = Field(default=False, description="True면 디버그 로그 활성화")

def festival_idea_entry(
    path: str = "public_agent/conversation.json",
    k: int = 2,
    speaker: Optional[str] = "어르신",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    max_chunks: int = 10,
    debug: bool = False,
) -> str:
    # 환경변수로도 강제 가능: FESTIVAL_IDEA_DEBUG=1
    debug = debug or os.getenv("FESTIVAL_IDEA_DEBUG") == "1"
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode on")

    try:
        blocks = load_conversation_json(path)
        flat = flatten_turns_to_text(blocks, speaker=speaker)
        if not flat.strip():
            logger.error("No usable conversation text after speaker filter")
            return json.dumps({"status": "error", "message": "No usable conversation text"}, ensure_ascii=False)
        chunks = chunk_text(flat, chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_chunks=max_chunks)
        return llm_extract_ideas(chunks, k=k, debug=debug)
    except Exception as e:
        logger.exception("festival_idea_entry failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

festival_idea_tool = StructuredTool.from_function(
    func=festival_idea_entry,
    name="festival_idea_tool",
    description=(
        "public_agent/conversation.json을 읽고 텍스트를 청킹한 뒤, 지역 어르신 대화에서 로컬 축제 아이디어 k개를 JSON으로 반환한다."
    ),
    args_schema=FestivalIdeaArgs,
)
