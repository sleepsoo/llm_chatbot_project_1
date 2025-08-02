import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import datetime

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

# 시간 정보 입력을 위해 함수 구현


def time_now():
    return datetime.datetime.now().strftime('%Y년%m월%d일 %H시%M분%S초')


# LLM 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 기존 대화 기록과 RAG를 결합한 응답생성 함수


def generate_response_with_memory(question, chat_history, retriever):
    # RAG
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 시간정보 가져오기
    current_time = time_now()

    # 시스템 프롬프트에 context와 대화 기억 지시
    messages = [
        ("system", f"""당신은 재테크 관련 지식이 풍부한 Question-Answering 챗봇입니다.
         주어진 컨텍스트와 이전 대화 내용을 모두 참고하여 사용자의 질문에 답변해주세요.(컨텍스트는 과거에 작성되어있으므로 오늘 날짜가 어떻게 되는지는 헷갈려하지 않기.)
         컨텍스트에서 날짜 확인이 가능하다면 가장 최근 컨텍스트를 활용하세요.
         이전 대화에서 사용자가 언급한 개인정보(나이, 상황 등), 관심분야 등을 기억하고 활용하세요.

         현재 시간: {current_time}
         사용자가 "지금 몇시야?", "오늘이 몇일이야?" 같은 시간 관련 질문을 하면 위의 현재 시간 정보를 활용하여 답변하세요.

         컨텍스트: {context}""")
    ]

    # 이전 대화 기록을 모두 메세지에 추가
    for q, a in chat_history:
        messages.append(("human", q))
        messages.append(("assistant", a))

    # 현재 질문 추가
    messages.append(("human", question))

    # LLM에 전체 컨텍스트 전달
    response = llm.invoke(messages)

    return response.content


# streamlit 구현
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
            # 대화 기록과 RAG 결합한 응답 생성
            answer = generate_response_with_memory(
                user_input,
                st.session_state["chat_history"],
                db.as_retriever()
            )

            st.session_state["chat_history"].append((user_input, answer))
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다.:{str(e)}")


# 대화 기록
st.subheader("your chat")
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
