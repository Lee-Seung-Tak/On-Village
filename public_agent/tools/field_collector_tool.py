from langchain.tools import BaseTool
from public_agent.tools.prompt_tool import VideoSpec
import json
from pydantic import PrivateAttr

class FieldCollectorTool(BaseTool):
    name: str = "field_collector_tool"
    description: str = "영상 제작 브리프 수집. 8개 필드를 대화로 채운다."

    # 🔒 pydantic-private 속성 선언
    _fields: dict = PrivateAttr()
    _descriptions: dict = PrivateAttr()
    _current_field: str = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fields = {f: None for f in VideoSpec.model_fields.keys()}
        self._descriptions = {
            f: VideoSpec.model_fields[f].description
            for f in VideoSpec.model_fields.keys()
        }
        self._current_field = "subject"

    def _run(self, query: str, run_manager=None) -> str:
        # 현재 필드 저장
        self._fields[self._current_field] = query

        # 다음 필드 찾기
        for f, v in self._fields.items():
            if v is None:
                self._current_field = f
                desc = self._descriptions[f] or f
                return f"좋습니다. 이제 **{f}**({desc})를 알려주세요."

        # 모든 필드 완료 시 JSON 반환
        return json.dumps(self._fields, ensure_ascii=False)

    async def _arun(self, query: str, run_manager=None) -> str:
        return self._run(query, run_manager)
