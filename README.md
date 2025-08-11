# On-Village

음성 Agent 및 생성형 Chatbot 프로젝트입니다.

- `nginx.conf` 및 `certbot` 폴더는 보안 문제로 Git에 올리지 않습니다.
- `nginx.conf`에서 초기 설정 시 HTTPS(443) 관련 설정은 주석 처리 후 인증서 신규 발급을 진행하고, 발급 완료 후 주석을 해제하고 서비스를 재실행해야 합니다.

## Project 실행 방법

1. Google Drive에서 받은 `nginx` 폴더를 프로젝트 루트 디렉토리에 저장합니다.
2. Google Drive에서 받은 `certbot` 폴더를 프로젝트 루트 디렉토리에 저장합니다.
3. Google Drive에서 받은 `setup.sh` 파일을 프로젝트 루트 디렉토리에 저장합니다.
4. 터미널에서 아래 명령어로 실행 권한을 부여합니다.
5. 아래 명령을 통하여 ec2환경에서 sudo 없이 docker를 실행할 수 있게 합니다.
```bash
sudo usermod -aG docker $USER
```
6. 아래의 명령을 통하여 세션을 갱신합니다.
```bash
newgrp docker 
```
7. docker-compose up -d 
```bash
chmod +x setup.sh
./setup.sh
```
