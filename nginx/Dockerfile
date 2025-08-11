# ./nginx/Dockerfile
FROM nginx:latest

# 기본 nginx 설정 파일 제거
RUN rm /etc/nginx/conf.d/default.conf

# 커스텀 설정 파일 복사
COPY nginx.conf /etc/nginx/conf.d/