import streamlit as st
import os
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

# 라이브러리 임포트
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# --- 0. 기본 설정 및 환경변수 ---
st.set_page_config(page_title="실버 가디언: 유튜브 AI 분석기", page_icon="🛡️")

# .env 파일 로드 (로컬 개발용)
load_dotenv()

# --- 1. State 정의 ---
class AgentState(TypedDict):
    youtube_url: str
    video_id: Optional[str]
    script_text: Optional[str]
    analysis_result: Optional[str]
    error: Optional[str]

# --- 2. 헬퍼 함수 ---
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

def get_video_script(video_id: str) -> str:
    """Video ID로 자막을 추출합니다."""
    try:
        ytt_api = YouTubeTranscriptApi()
        
        transcript_list = ytt_api.list(video_id)
        
        # 자막 우선순위: 한국어 수동 -> 영어 수동 -> 한국어 자동 -> 영어 자동
        try:
            transcript = transcript_list.find_transcript(['ko'])
        except:
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                try:
                    transcript = transcript_list.find_generated_transcript(['ko'])
                except:
                    # 최후의 수단: 번역 가능한 자막을 한국어로 번역
                    try:
                        transcript = transcript_list.find_manually_created_transcript(['en'])
                        transcript = transcript.translate('ko')
                    except:
                         # 자동 생성된 영어라도 가져와서 번역 시도
                        transcript = transcript_list.find_generated_transcript(['en'])
                        transcript = transcript.translate('ko')

        formatter = TextFormatter()
        script_text = formatter.format_transcript(transcript.fetch())
        return script_text.replace("\n", " ")
        
    except (TranscriptsDisabled, NoTranscriptFound):
        return "ERROR: 이 영상에는 자막이 없습니다."
    except Exception as e:
        return f"ERROR: 자막 추출 실패 ({str(e)})"

# --- 3. 노드 함수 정의 ---
def script_loader_node(state: AgentState):
    """URL에서 스크립트 추출"""
    url = state['youtube_url']
    video_id = extract_video_id(url)
    
    if not video_id:
        return {"error": "유효하지 않은 유튜브 URL입니다."}

    script = get_video_script(video_id)
    
    if script.startswith("ERROR"):
        return {"error": script, "script_text": None}
    
    return {"video_id": video_id, "script_text": script}

def text_analysis_node(state: AgentState):
    """Gemini를 사용하여 텍스트 분석"""
    if state.get("error"):
        return {"analysis_result": f"분석 불가: {state['error']}"}
        
    script = state['script_text']
    
    # API 키 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "Google API Key가 설정되지 않았습니다."}

    # 모델 초기화 (참고: gemini-2.5-flash는 예시 모델명이며, 실제 사용 가능한 모델명으로 변경 필요할 수 있음. 예: gemini-1.5-flash)
    # 사용자가 요청한 모델명 유지, 필요시 'gemini-1.5-flash'로 변경하세요.
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
        
        prompt_text = f"""
        당신은 노인 소비자 보호 및 금융 사기 예방 전문가입니다.
        아래 텍스트(유튜브 스크립트 등)를 정밀 분석하여, 판단력이 흐려지기 쉬운 고령층을 타깃으로 한 '불법 투자 권유', '기만적 상품 판매', 또는 '스팸성 콘텐츠'인지 판별하세요.

        [분석할 스크립트]
        "{script[:10000]}" ... (길이 제한으로 일부 생략)

        [중점 분석 항목]
        1. **심리적 조작 및 공포 마케팅**: 건강 공포심 유발, 거짓 긴급성 강조.
        2. **비현실적 약속 및 과장 광고**: 원금 보장, 기적의 치료법 등.
        3. **위험한 행동 유도**: 리딩방 유입, 특정 물품 구매 강요.

        [최종 답변 형식]
        ## 🚨 노인 대상 유해 콘텐츠 분석 결과

        **1. 판정**: [고위험 스팸 및 사기 의심 / 주의 필요(과장 광고) / 안전한 콘텐츠]
        **2. 위험도 점수**: [0~100점]
        
        **3. 주요 적발 소견**:
           - **[자극적 키워드]**:
           - **[심리 조작 기법]**:
           - **[유도 방식]**:

        **4. 소비자 행동 지침**:
           - (구체적인 행동 가이드)
        """
        
        response = llm.invoke([HumanMessage(content=prompt_text)])
        return {"analysis_result": response.content}
    except Exception as e:
        return {"error": f"AI 분석 중 오류 발생: {str(e)}"}

# --- 4. 그래프 구축 ---
def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("loader", script_loader_node)
    workflow.add_node("analyst", text_analysis_node)
    workflow.set_entry_point("loader")
    workflow.add_edge("loader", "analyst")
    workflow.add_edge("analyst", END)
    return workflow.compile()

# --- 5. Streamlit UI 구성 ---
def main():
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        # API 키 입력 (환경변수에 없으면 입력받음)
        if not os.getenv("GOOGLE_API_KEY"):
            api_key_input = st.text_input("Google API Key 입력", type="password")
            if api_key_input:
                os.environ["GOOGLE_API_KEY"] = api_key_input
        
        st.info("이 앱은 YouTube 자막을 추출하여 Google Gemini로 사기/과장 광고 여부를 분석합니다.")

    st.title("🛡️ 실버 가디언")
    st.subheader("어르신을 위한 유튜브 유해 콘텐츠 탐지기")

    # URL 입력
    url = st.text_input("분석할 유튜브 영상 주소를 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("🔍 분석 시작", type="primary"):
        if not url:
            st.warning("URL을 입력해주세요.")
            return

        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Google API Key가 필요합니다. 사이드바에서 입력하거나 .env 파일을 설정해주세요.")
            return

        # 그래프 실행
        app = create_graph()
        
        # 진행 상황 표시용 컨테이너
        status_container = st.container()
        
        with st.spinner("영상을 분석하고 있습니다... (자막 추출 및 AI 분석)"):
            try:
                inputs = {"youtube_url": url}
                result = app.invoke(inputs)
                
                # 에러 처리
                if result.get("error"):
                    st.error(f"오류 발생: {result['error']}")
                else:
                    # 결과 표시
                    st.success("분석이 완료되었습니다!")
                    
                    # 1. 영상 썸네일 표시
                    if result.get("video_id"):
                        st.image(f"https://img.youtube.com/vi/{result['video_id']}/0.jpg", width=400)
                    
                    # 2. 분석 결과 (Markdown)
                    st.markdown("---")
                    st.markdown(result["analysis_result"])
                    
                    # 3. 추출된 스크립트 (Expander로 숨김 처리)
                    with st.expander("📝 추출된 자막 원본 보기"):
                        st.text_area("자막 내용", result.get("script_text", ""), height=300)
                        
            except Exception as e:
                st.error(f"실행 중 예기치 못한 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()