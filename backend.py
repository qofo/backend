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

# 환경 변수 로드
load_dotenv()

# --- 0. FastAPI 설정 (변수명: app) ---
app = FastAPI(
    title="YouTube Scam Detector API",
    description="유튜브 영상의 자막을 분석하여 노인 대상 사기/스팸 여부를 판별합니다.",
    version="1.0.0"
)

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

# --- 3. 헬퍼 함수 ---
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

def get_video_script(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        
        try:
            transcript = transcript_list.find_transcript(['ko', 'en'])
        except:
            pass

        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['ko', 'en'])
            except:
                pass
        
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['en']) 
                transcript = transcript.translate('ko')
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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt_text = f"""
    당신은 노인 소비자 보호 및 금융 사기 예방 전문가입니다.
    아래 텍스트(유튜브 스크립트 등)를 정밀 분석하여, 판단력이 흐려지기 쉬운 고령층을 타깃으로 한 '불법 투자 권유', '기만적 상품 판매', 또는 '스팸성 콘텐츠'인지 판별하세요.

    [분석할 스크립트]
    "{script[:5000]}" ... (이하 생략)

    [중점 분석 항목]
    1. **심리적 조작 및 공포 마케팅 (Fear & Greed)**
    2. **비현실적 약속 및 과장 광고**
    3. **위험한 행동 유도 (Call to Action)**

    [최종 답변 형식]
    ## 🚨 노인 대상 유해 콘텐츠 분석 결과

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

# --- 5. 그래프 구축 ---
workflow = StateGraph(AgentState)
workflow.add_node("loader", script_loader_node)
workflow.add_node("analyst", text_analysis_node)
workflow.set_entry_point("loader")
workflow.add_edge("loader", "analyst")
workflow.add_edge("analyst", END)

# [중요] 변수명을 graph_runner로 변경하여 FastAPI의 app과 충돌 방지
graph_runner = workflow.compile()

# --- 6. API 엔드포인트 ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video_endpoint(request: AnalyzeRequest):
    try:
        initial_state = {"youtube_url": request.youtube_url}
        # 여기서 graph_runner를 실행합니다.
        result = await graph_runner.ainvoke(initial_state)
        
        if result.get("error"):
            return AnalyzeResponse(
                video_id=result.get("video_id"),
                error=result.get("error"),
                analysis_result=result.get("analysis_result")
            )
            
        return AnalyzeResponse(
            video_id=result.get("video_id"),
            analysis_result=result.get("analysis_result"),
            error=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    # 파일명이 new_main.py라고 가정합니다.
    # 만약 파일명이 다르다면 "파일명:app" 으로 수정해야 합니다.
    uvicorn.run("new_main:app", host="127.0.0.1", port=8000, reload=True)