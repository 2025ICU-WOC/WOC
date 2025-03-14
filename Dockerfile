# 베이스 이미지 설정
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 앱 코드 복사
COPY . .

# 컨테이너 포트 설정
EXPOSE 5000

# 실행 명령어
CMD ["python", "app.py"]