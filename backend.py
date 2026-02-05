import os
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 자막 추출 라이브러리
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter

# LangChain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# youtube 검색을 위한 라이브러리
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

# --- 2. State 정의 ---
class AgentState(TypedDict):
    youtube_url: str
    video_id: Optional[str]
    script_text: Optional[str]
    analysis_result: Optional[str]
    error: Optional[str]

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


# 비디오 url에서 video_id 추출하기
def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

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

# video_id에서 스크립트 추출하기
def get_video_script(video_id: str) -> str:
    try:
        #transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_list = YouTubeTranscriptApi.list(video_id)
        transcript = None
        
        try:
            transcript = transcript_list.find_transcript(['ko', 'en'])
        except:
            pass

        if not transcript: # find_transcript 함수가 없는 경우
            try:
                transcript = transcript_list.find_generated_transcript(['ko', 'en'])
            except:
                pass
        
        if not transcript: # 한국어가 없는 경우
            try:
                transcript = transcript_list.find_generated_transcript(['en']) 
                transcript = transcript.translate('ko') # 한국어를 영어로 번역?? 
                # [ TODO ]: 한국어로 번역할지, 그냥 쓸지 정해야 함
            except:
                for t in transcript_list:
                    transcript = t.translate('ko')
                    break

        if not transcript:
             return "ERROR: 적절한 자막을 찾을 수 없습니다."

        formatter = TextFormatter()
        script_text = formatter.format_transcript(transcript.fetch())
        return script_text.replace("\n", " ")
        
    except (TranscriptsDisabled, NoTranscriptFound):
        return "ERROR: 이 영상에는 자막이 없습니다."
    except Exception as e:
        return f"ERROR: 자막 추출 실패 ({str(e)})"

# --- 4. 노드 함수 ---
def script_loader_node(state: AgentState):
    url = state['youtube_url']
    video_id = extract_video_id(url)
    if not video_id:
        return {"error": "유효하지 않은 유튜브 URL입니다."}

    script = get_video_script(video_id)
    if script.startswith("ERROR"):
        return {"error": script, "script_text": None}
    
    return {"video_id": video_id, "script_text": script}

async def text_analysis_node(state: AgentState):
    if state.get("error"):
        return {"analysis_result": f"분석 불가: {state['error']}"}
        
    script = state['script_text']
    # Gemini 모델 설정
    # [TODO] : 모델을 무엇으로 할지 정해야 함
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # [TODO] : 프롬프트 수정해야 함
    prompt_text = f"""
    당신은 소비자 보호 및 금융 사기 예방 전문가입니다.
    아래 텍스트(유튜브 스크립트 등)를 정밀 분석하여, '불법 투자 권유', '기만적 상품 판매', 또는 '스팸성 콘텐츠'인지 판별하세요.

    [분석할 스크립트]
    "{script[:5000]}" ... (이하 생략)

    [중점 분석 항목]
    1. **심리적 조작 및 공포 마케팅 (Fear & Greed)**
    2. **비현실적 약속 및 과장 광고**
    3. **위험한 행동 유도 (Call to Action)**

    [최종 답변 형식]
    ## 🚨 허위 광고 등 유해 콘텐츠 분석 결과

    **1. 판정**: [고위험 스팸 및 사기 의심 / 주의 필요(과장 광고) / 안전한 콘텐츠]
    **2. 위험도 점수**: [0~100점] (점수가 높을수록 위험)

    **3. 주요 적발 소견**:
       - **[자극적 키워드]**:
       - **[심리 조작 기법]**:
       - **[유도 방식]**:

    **4. 최종 요약**:
    """

    response = await llm.ainvoke([HumanMessage(content=prompt_text)])
    return {"analysis_result": response.content}

# 랭그래프 구축하기
workflow = StateGraph(AgentState)
workflow.add_node("loader", script_loader_node)
workflow.add_node("analyst", text_analysis_node)
workflow.set_entry_point("loader")
workflow.add_edge("loader", "analyst")
workflow.add_edge("analyst", END)

graph_runner = workflow.compile()

from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        graph_runner.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

# 프론트엔드가 호출하는 부분
@app.post("/search", response_model=SearchResponse)
async def search_video_endpoint(request: SearchRequest):
    """
    제목과 채널명을 받아 유튜브 URL을 검색하고, 
    해당 영상의 자막을 추출하여 즉시 사기 여부를 분석합니다.
    """
    # 1. 검색어 조합
    query = f"{request.title} {request.channel}".strip()
    print(f"검색 요청: '{query}'")
    
    # 2. 실제 YouTube 검색 수행
    search_result = search_video_on_youtube(query)
    
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
    
    initial_state = {"youtube_url": search_result['url']}
    analysis_output = await graph_runner.ainvoke(initial_state)
    
    # 4. 결과 통합 반환
    return SearchResponse(
        video_id=search_result['video_id'],
        youtube_url=search_result['url'],
        title=search_result['title'],
        channel_title=search_result['channel_title'],
        found=True,
        message="검색 및 분석 완료",
        analysis_result=analysis_output.get("analysis_result"),
        error=analysis_output.get("error")
    )

# # --- 6. API 엔드포인트 ---
# @app.post("/analyze", response_model=AnalyzeResponse)
# async def analyze_video_endpoint(request: AnalyzeRequest):
#     try:
#         initial_state = {"youtube_url": request.youtube_url}
#         # 여기서 graph_runner를 실행합니다.
#         result = await graph_runner.ainvoke(initial_state)
        
#         if result.get("error"):
#             return AnalyzeResponse(
#                 video_id=result.get("video_id"),
#                 error=result.get("error"),
#                 analysis_result=result.get("analysis_result")
#             )
            
#         return AnalyzeResponse(
#             video_id=result.get("video_id"),
#             analysis_result=result.get("analysis_result"),
#             error=None
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    # 파일명이 new_main.py라고 가정합니다.
    # 만약 파일명이 다르다면 "파일명:app" 으로 수정해야 합니다.
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)