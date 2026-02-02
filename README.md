# youtube-spam-check
유튜브 스팸 확인 데모 프로토타입
# 설치 및 설정 (Setup)
아래 단계에 따라 프로젝트를 로컬 환경에 설치하고 실행하세요.

1. 가상환경 구축 및 실행
먼저 프로젝트 루트 폴더에서 Python 가상환경(venv)을 생성하고 활성화합니다.

Windows:

```sh
python -m venv venv
venv/Scripts/activate
```

macOS/Linux:

```sh
python3 -m venv venv
source venv/bin/activate
```
2. 환경 변수 설정
프로젝트 루트 디렉토리에 .env 파일을 생성하고 본인의 Google API 키를 입력합니다.

Plaintext
# .env 파일 생성 후 아래 내용 입력
```
GOOGLE_API_KEY=your_actual_api_key_here
```
3. 필수 라이브러리 설치
requirements.txt 파일에 명시된 의존성 패키지들을 설치합니다.

```sh
pip install -r requirements.txt
```
# 실행 방법 (Execution)
터미널에서 실행하기
콘솔 환경에서 로직을 확인하려면 아래 명령어를 입력하세요.

```sh
python main.py
```
웹 인터페이스로 실행하기 (Streamlit)
브라우저 기반의 UI를 사용하려면 Streamlit을 통해 실행합니다.

```sh
streamlit run app.py
```
