import os
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi


# 자막 추출 라이브러리
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# LangChain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# 환경 변수 로드 (GOOGLE_API_KEY 필수)
load_dotenv()

# --- 1. State 정의 ---
class AgentState(TypedDict):
    youtube_url: str
    video_id: Optional[str]
    script_text: Optional[str]    # 추출된 스크립트
    analysis_result: Optional[str]
    error: Optional[str]

# --- 2. 헬퍼 함수 (자막 추출) ---

def extract_video_id(url: str) -> Optional[str]:
    """유튜브 URL에서 Video ID 추출"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# 상단 imports 부분은 그대로 두되, 함수 내 import는 제거
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_video_script(video_id: str) -> str:
    """
    Video ID로 자막을 추출합니다. (한국어 -> 영어 -> 자동생성 순)
    """
    try:
        # 1. 자막 리스트 가져오기
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        #transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # 2. 자막 찾기 (filter를 사용하면 try-except 중첩을 줄일 수 있습니다)
        # 'ko' 수동 -> 'en' 수동 -> 'ko' 자동 -> 'en' 자동 순서로 찾습니다.
        try:
            transcript = transcript_list.find_transcript(['ko', 'en'])
        except:
            # 수동 자막이 없으면 자동 생성 자막 탐색
            try:
                transcript = transcript_list.find_generated_transcript(['ko', 'en'])
            except:
                 # 그래도 없으면 번역 가능한 아무 언어나 가져와서 한국어로 번역 시도
                transcript = transcript_list.find_manually_created_transcript(['en'])
                transcript = transcript.translate('ko')

        # 3. 텍스트로 변환
        formatter = TextFormatter()
        script_text = formatter.format_transcript(transcript.fetch())
        
        return script_text.replace("\n", " ")
        
    except (TranscriptsDisabled, NoTranscriptFound):
        return "ERROR: 이 영상에는 자막이 없습니다."
    except Exception as e:
        return f"ERROR: 자막 추출 실패 ({str(e)})"

# --- 3. 노드 함수 정의 ---

def script_loader_node(state: AgentState):
    """URL에서 스크립트만 빠르게 추출하는 노드"""
    url = state['youtube_url']
    print(f"📥 스크립트 추출 시도 중... ({url})")
    
    video_id = extract_video_id(url)
    if not video_id:
        return {"error": "유효하지 않은 유튜브 URL입니다."}

    script = get_video_script(video_id)
    
    if script.startswith("ERROR"):
        return {"error": script, "script_text": None}
    
    print(f"✅ 스크립트 추출 완료 (길이: {len(script)}자)")
    return {"video_id": video_id, "script_text": script}

def text_analysis_node(state: AgentState):
    """Gemini를 사용하여 텍스트 패턴을 분석하는 노드"""
    if state.get("error"):
        return {"analysis_result": f"분석 불가: {state['error']}"}
        
    script = state['script_text']
    print("🤖 AI 텍스트 포렌식 분석 중 (Gemini 2.0 Flash)...")

    # 모델 초기화
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 프롬프트: 텍스트 기반 AI 판별에 집중
    prompt_text = f"""
    당신은 노인 소비자 보호 및 금융 사기 예방 전문가입니다.
아래 텍스트(유튜브 스크립트 등)를 정밀 분석하여, 판단력이 흐려지기 쉬운 고령층을 타깃으로 한 '불법 투자 권유', '기만적 상품 판매', 또는 '스팸성 콘텐츠'인지 판별하세요.

[분석할 스크립트]
"{script[:5000]}" ... (이하 생략)

[중점 분석 항목]
1. **심리적 조작 및 공포 마케팅 (Fear & Greed)**:
   - "병원에서도 알려주지 않는", "지금 모르면 큰일 나는" 등 건강에 대한 과도한 공포심 유발.
   - "정부 지원금 소멸 예정", "마감 임박" 등 거짓 긴급성을 강조하여 이성적 판단 방해.
   - "자식에게 짐이 되지 않으려면", "노후 파산" 등 노인 빈곤/고립 심리를 악용하는 멘트.

2. **비현실적 약속 및 과장 광고**:
   - "원금 100% 보장", "무조건 오르는 종목", "기적의 치료법" 등 확정적 단어 사용.
   - 구체적인 근거 없이 "비밀 정보", "세력 매집주"라며 정보의 희소성을 가장.
   - 제도권 금융기관이나 공공기관을 사칭하거나 모호하게 연관 지어 신뢰를 날조.

3. **위험한 행동 유도 (Call to Action)**:
   - "무료 리딩방 입장", "상담 번호로 문자 전송", "고정 댓글 링크 클릭" 등 외부 채널 유입 강요.
   - 영상 내용과 무관한 특정 건강식품, 코인, 비상장 주식 등의 구매 유도.

[최종 답변 형식]
## 🚨 노인 대상 유해 콘텐츠 분석 결과

**1. 판정**: [고위험 스팸 및 사기 의심 / 주의 필요(과장 광고) / 안전한 콘텐츠]
**2. 위험도 점수**: [0~100점] (점수가 높을수록 위험)

**3. 주요 적발 소견**:
   - **[자극적 키워드]**: (스크립트 내 "원금 보장", "기적의 효능" 등 문제 발언 직접 인용)
   - **[심리 조작 기법]**: (어르신들의 불안감을 어떻게 조장했는지 분석)
   - **[유도 방식]**: (카카오톡방, 전화번호 수집 등 구체적인 유도 패턴 지적)

**4. 소비자 행동 지침**:
   - (이 콘텐츠를 접한 노인 사용자가 취해야 할 구체적인 행동 가이드. 예: "절대 링크를 누르지 마세요", "자녀와 상의하세요")
    """

    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {"analysis_result": response.content}

# --- 4. 그래프 구축 ---

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("loader", script_loader_node)
workflow.add_node("analyst", text_analysis_node)

# 엣지 연결
workflow.set_entry_point("loader")
workflow.add_edge("loader", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

# --- 5. 실행부 ---
if __name__ == "__main__":
    test_url = input("분석할 유튜브 링크 입력: ")
    
    inputs = {"youtube_url": test_url}
    result = app.invoke(inputs)
    
    print("\n" + "="*40)
    print(result["analysis_result"])
    print("="*40)