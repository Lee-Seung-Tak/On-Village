# public_agent/tools/video_tool.py
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateVideosConfig
from google.cloud import storage
from public_agent.prompts.prompt import build_prompt

load_dotenv()


def _outputs_dir() -> str:
    """outputs 디렉토리 보장"""
    base = os.path.dirname(os.path.dirname(__file__))  # /public_agent
    out = os.path.join(base, "outputs")
    os.makedirs(out, exist_ok=True)
    return out


def _enforce_cartoon_human(style: str, negative_prompt: str, composition: str):
    """
    '사람 등장 + 카툰/애니'를 보장하고, 실사 느낌을 억제하기 위해
    style/negative_prompt/composition을 보정한다. (중복 삽입 방지)
    """
    s = (style or "").strip()
    n = (negative_prompt or "").strip()
    c = (composition or "").strip()

    s_low, n_low, c_low = s.lower(), n.lower(), c.lower()

    # style: cartoon/animation + human characters + cel-shaded
    add_style = []
    if ("cartoon" not in s_low) and ("animation" not in s_low) and ("애니" not in s_low):
        add_style.append("cartoon / animation")
    if ("human characters" not in s_low) and ("human" not in s_low) and ("사람" not in s_low):
        add_style.append("human characters")
    if "cel-shaded" not in s_low and "셀셰이딩" not in s_low:
        add_style.append("cel-shaded look")
    if add_style:
        s = (s + (", " if s else "") + ", ".join(add_style)).strip()

    # negative: 실사/포토리얼 억제
    neg_tokens = [
        "photorealistic", "live-action", "hyper-realistic",
        "uncanny", "realistic skin", "film camera"
    ]
    add_negs = [t for t in neg_tokens if t not in n_low]
    if add_negs:
        n = (n + (", " if n else "") + ", ".join(add_negs)).strip()

    # composition: 인물 명시(중/근접 샷 혼합)
    if all(k not in c_low for k in ["human", "people", "person", "사람", "인물"]):
        c = (c + (", " if c else "") +
             "character-centered shots with clearly visible human figures, mix of medium and close shots"
             ).strip()

    return s, n, c


def generate_video(
    subject: str,
    context: str,
    action: str,
    style: str,
    camera_motion: str,
    composition: str,
    ambiance: str,
    negative_prompt: str,
) -> str:
    """
    모든 필드를 받아 프롬프트를 만들고 Google GenAI(Veo)로 영상 생성.
    성공: {"status":"success","gcs_uri":...,"local_file":...,"prompt":...}
    실패: {"status":"error"}
    """
    try:
        # ENV
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION", "us-central1")
        model = os.getenv("GOOGLE_MODEL", "veo-3.0-generate-preview")
        output_gcs_uri = os.getenv("OUTPUT_GCS_URI")  # ex) gs://bucket/path/
        cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred

        # ✅ 카툰+사람 강제 보정 (툴 레벨에서 항상 보장)
        style, negative_prompt, composition = _enforce_cartoon_human(
            style, negative_prompt, composition
        )

        # Prompt
        final_prompt = build_prompt(
            subject=subject,
            context=context,
            action=action,
            style=style,
            camera_motion=camera_motion,
            composition=composition,
            ambiance=ambiance,
            negative_prompt=negative_prompt,
        )

        # Client & request
        client = genai.Client(project=project_id, location=location)
        operation = client.models.generate_videos(
            model=model,
            prompt=final_prompt,
            config=GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=8,
                output_gcs_uri=output_gcs_uri,
            ),
        )

        # Polling (최대 10분)
        start = time.time()
        while not operation.done:
            if time.time() - start > 600:
                return json.dumps({"status": "error"}, ensure_ascii=False)
            time.sleep(10)
            operation = client.operations.get(operation)

        # GCS URI
        uri = operation.result.generated_videos[0].video.uri  # gs://bucket/path/xxx.mp4

        # Download to outputs/
        storage_client = storage.Client()
        bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(blob_path)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_name = f"{ts}_{os.path.basename(blob_path)}"
        local_path = os.path.join(_outputs_dir(), local_name)
        blob.download_to_filename(local_path)

        return json.dumps(
            {
                "status": "success",
                "gcs_uri": uri,
                "local_file": local_path,
                "prompt": final_prompt,
            },
            ensure_ascii=False,
        )
    except Exception:
        return json.dumps({"status": "error"}, ensure_ascii=False)
