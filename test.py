import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# API KEY 정보 로드
load_dotenv()

# 벡터스토어 불러오기


@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024
    )
    db = FAISS.load_local("my_faiss_index",
                          embeddings,
                          allow_dangerous_deserialization=True)
    return db


db = load_vectorstore()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 대화 기록과 RAG를 결합한 응답 생성 함수


def generate_response_with_memory(question, chat_history, retriever):
    # 1. RAG: 관련 문서 검색
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. 대화 히스토리를 메시지 형태로 구성
    messages = [
        ("system", f"""당신은 재테크 관련 지식이 풍부한 Question-Answering 챗봇입니다.

        주어진 컨텍스트와 이전 대화 내용을 모두 참고하여 사용자의 질문에 답변해주세요.
        이전 대화에서 사용자가 언급한 개인정보(나이, 상황 등)를 기억하고 활용하세요.

        컨텍스트:
        {context}""")
    ]

    # 3. 이전 대화 기록 추가
    for q, a in chat_history:
        messages.append(("human", q))
        messages.append(("assistant", a))

    # 4. 현재 질문 추가
    messages.append(("human", question))

    # 5. LLM 호출
    response = llm.invoke(messages)

    return response.content


# Streamlit 구현
st.title("🏦💸재테크 전략 도우미 Chatbot🧠")

# 대화 히스토리를 session state에 저장
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 대화 기록 삭제
if st.button("대화 내역 삭제", type="primary"):
    st.session_state["chat_history"] = []
    st.rerun()

# 채팅 실행 버튼(Enter 키 가능)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "메세지를 입력하세요", key="user_input", placeholder="메세지 입력 후 Enter")
    submitted = st.form_submit_button("전송")

if submitted and user_input.strip():
    with st.spinner("답변을 생성하는 중..."):
        try:
            # 대화 기록과 RAG를 결합한 응답 생성
            answer = generate_response_with_memory(
                user_input,
                st.session_state["chat_history"],
                db.as_retriever()
            )

            # 대화 기록에 추가
            st.session_state["chat_history"].append((user_input, answer))

        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")

# 대화 기록
st.subheader("대화 기록")
for i, (q, a) in enumerate(reversed(st.session_state["chat_history"])):
    real_idx = len(st.session_state["chat_history"]) - 1 - i
    col1, col2 = st.columns([7, 1])
    with col1:
        st.markdown(f"**질문:** {q}")
        st.markdown(f"**답변:** {a}")
    with col2:
        if st.button("🗑️", key=f"delete_{real_idx}"):
            st.session_state["chat_history"].pop(real_idx)
            st.rerun()
    st.markdown("---")

# 디버그용: 현재 저장된 대화 기록 확인
if st.checkbox("디버그: 저장된 대화 기록 보기"):
    st.write("현재 저장된 대화 수:", len(st.session_state["chat_history"]))
    for i, (q, a) in enumerate(st.session_state["chat_history"]):
        st.write(f"{i+1}. Q: {q[:50]}...")
        st.write(f"   A: {a[:50]}...")
