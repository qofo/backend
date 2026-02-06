from typing import TypedDict, Optional
import re # 정규표현식용

# LangChain & LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# 자막 추출 라이브러리
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter


# State 정의
class AgentState(TypedDict):
    youtube_url: str
    video_id: Optional[str]
    script_text: Optional[str]
    analysis_result: Optional[str]
    error: Optional[str]

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

# video_id에서 스크립트 추출하기
def get_video_script(video_id: str) -> str:
    try:
        #transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
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

def text_analysis_node(state: AgentState):
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

    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {"analysis_result": response.content}


def build_graph():
    # 랭그래프 구축하기
    workflow = StateGraph(AgentState)
    workflow.add_node("loader", script_loader_node)
    workflow.add_node("analyst", text_analysis_node)
    workflow.set_entry_point("loader")
    workflow.add_edge("loader", "analyst")
    workflow.add_edge("analyst", END)

    return workflow.compile()
