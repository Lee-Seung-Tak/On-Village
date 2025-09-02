# public_agent/registry.py

from public_agent.tools.festival_idea_tool import festival_idea_tool
from public_agent.tools.prompt_tool import fill_video_fields_tool
from public_agent.tools.video_tool import generate_video
from langchain.tools import StructuredTool
from public_agent.tools.field_collector_tool import FieldCollectorTool
from pydantic import BaseModel, Field


# video_tool용 schema 정의 (generate_video 함수와 변수명 동일)
class GenerateVideoArgs(BaseModel):
    subject: str = Field(..., description="주 피사체/인물/대상")
    context: str = Field(..., description="배경/장소/상황")
    action: str = Field(..., description="인물의 대사/행동")
    style: str = Field(..., description="시각적 스타일")
    camera_motion: str = Field(..., description="카메라 무브/샷 종류")
    composition: str = Field(..., description="구도/렌즈/프레이밍")
    ambiance: str = Field(..., description="분위기/조명/사운드")
    negative_prompt: str = Field(..., description="피해야 하는 요소")

def video_tool_entry(**kwargs) -> str:
    return generate_video(**kwargs)

video_generation_tool = StructuredTool.from_function(
    func=video_tool_entry,
    name="video_generation_tool",
    description="8개 필드를 받아 Google GenAI(Veo)로 영상을 생성",
    args_schema=GenerateVideoArgs,
)

# 사용할 모든 도구들을 한 리스트에 모음
tools = [
    festival_idea_tool,
    fill_video_fields_tool,
    video_generation_tool,
    FieldCollectorTool(),
]
