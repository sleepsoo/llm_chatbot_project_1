import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# os.environ["OPENAI_API_KEY"] = ''


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

# streamlit 구현
st.title("🏦💸재테크 전략 도우미 Chatbot🧠")

# 대화 히스토리를 session state에 저장
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 대화 기록 삭제
if st.button("대화 내역 삭제", type="primary"):
    st.session_state["chat_history"] = []

# 채팅 실행 버튼(Enter 키 가능)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "메세지를 입력하세요", key="user_input", placeholder="메세지 입력 후 Enter")
    submitted = st.form_submit_button("전송")

if submitted and user_input.strip():
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4.1-mini"),
        retriever=db.as_retriever()
    )
    with st.spinner("답변을 생성하는 중..."):
        answer = qa.run(user_input)
    st.session_state["chat_history"].append((user_input, answer))


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
