# 베이스 이미지
FROM python:3.11

ENV PYTHONUNBUFFERED=1

# 작업 디렉토리
WORKDIR /app

# pip upgrade
RUN python -m pip install --upgrade pip

# 의존성 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade chromadb
# 현재 디렉토리 전체를 /app으로 복사
COPY . .

# 포트 노출
EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]