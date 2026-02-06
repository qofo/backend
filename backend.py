import os

from typing import Optional
from dotenv import load_dotenv

# fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# youtube 검색을 위한 라이브러리
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 사용자 정의 모듈
from supervisor import build_graph


# 환경 변수 로드
load_dotenv()

# FastAPI 설정
app = FastAPI(
    title="YouTube Scam Detector API",
    description="유튜브 영상의 자막을 분석하여 노인 대상 사기/스팸 여부를 판별합니다.",
    version="1.0.0"
)

# [TODO]: 네트워크 접근 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. 데이터 모델 (Pydantic) ---
class AnalyzeRequest(BaseModel):
    youtube_url: str

class AnalyzeResponse(BaseModel):
    video_id: Optional[str] = None
    analysis_result: Optional[str] = None
    error: Optional[str] = None



# 프론트엔드가 요청하는 내용
class SearchRequest(BaseModel):
    title: str
    channel: str
    # [TODO] : runtime 추가

# 백엔드가 최종적으로 주는 내용
class SearchResponse(BaseModel):
    video_id: Optional[str] = None
    youtube_url: Optional[str] = None
    title: Optional[str] = None
    channel_title: Optional[str] = None
    found: bool # TODO 필요한지 
    message: Optional[str] = None
    analysis_result: Optional[str] = None # 분석 결과
    error: Optional[str] = None # 분석 중 발생한 에러


# 제목, 채널 명으로 유튜브 링크 검색하기
def search_video_on_youtube(query: str):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {"error": "서버 설정 오류: YOUTUBE_API_KEY가 없습니다."}

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        # 검색 요청 (type='video', part='snippet', 결과 1개)
        search_response = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=1
        ).execute()

        items = search_response.get("items", [])
        if not items:
            return None

        item = items[0]
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        channel_title = item["snippet"]["channelTitle"]


        return {
            "video_id": video_id,
            "title": title,
            "channel_title": channel_title,
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }

    except HttpError as e:
        return {"error": f"YouTube API 오류: {e}"}
    except Exception as e:
        return {"error": f"검색 중 오류 발생: {e}"}



# 프론트엔드가 호출하는 부분
@app.post("/search", response_model=SearchResponse)
def search_video_endpoint(request: SearchRequest):
    """
    제목과 채널명을 받아 유튜브 URL을 검색하고, 
    해당 영상의 자막을 추출하여 즉시 사기 여부를 분석합니다.
    """
    # 백엔드: 영상 URL 수집 및 전송
    # 1. 검색어 조합
    query = f"{request.title} {request.channel}".strip()
    print(f"검색 요청: '{query}'")
    
    # 2. 실제 YouTube 검색 수행
    search_result = search_video_on_youtube(query)
    # print('로그 : 유튜브 api 실행 완료')
    
    # 2-1. 검색 에러 처리
    if search_result and "error" in search_result:
         return SearchResponse(
            found=False,
            message=search_result["error"]
        )

    # 2-2. 검색 결과 없음 처리
    if not search_result:
        return SearchResponse(found=False, message="영상을 찾을 수 없습니다.")
    
    # 3. 검색된 URL로 분석(AgentGraph) 실행
    print(f"검색 성공: {search_result['title']} ({search_result['url']}) -> 분석 시작")

    # [TODO]:  DB에 기존 데이터가 있는가?
    IS_IN_DB = False
    if IS_IN_DB:
        pass
    else:
        # [TODO]: 백엔드: DB에 저장, AI 결과는 Null로

        # [TODO]: 백엔드: 데이터 검증(FE 데이터 vs BE 데이터)
        try:
            # 슈퍼바이저: 데이터 전송 및 리포트 생성
            initial_state = {"youtube_url": search_result['url']}
            analysis_output = graph_runner.invoke(initial_state)
            print("분석 결과: ", analysis_output.get("analysis_result"))

            # [TODO]: 백엔드: DB에 리포트 내용 업데이트 저장
            
            # 결과 통합 반환
            if analysis_output.get("error"):
                return SearchResponse(
                video_id=search_result['video_id'],
                youtube_url=search_result['url'],
                title=search_result['title'],
                channel_title=search_result['channel_title'],
                found=True,
                message="검색 성공 및 분석 실패",
                error=analysis_output.get("error")
            )

            return SearchResponse(
                video_id=search_result['video_id'],
                youtube_url=search_result['url'],
                title=search_result['title'],
                channel_title=search_result['channel_title'],
                found=True,
                message="검색 및 분석 완료",
                analysis_result=analysis_output.get("analysis_result"),
                error=None
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}


graph_runner = build_graph()
if __name__ == "__main__":
    import uvicorn

    

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)