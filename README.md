# 🏦 재테크 전략 도우미 RAG Chatbot
**메타코드 AI LLM 부트캠프 최종 프로젝트**

<p align="left">
  <img src="[https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)"/>
  <img src="[https://img.shields.io/badge/LangChain-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white](https://img.shields.io/badge/LangChain-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white)"/>
  <img src="[https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)"/>
  <img src="[https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)"/>
  <img src="[https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)"/>
  <img src="[https://img.shields.io/badge/BeautifulSoup-306998?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/BeautifulSoup-306998?style=for-the-badge&logo=python&logoColor=white)"/>
</p>

다중 플랫폼 크롤링과 OCR을 활용하여 흩어진 재테크 정보를 통합하고, 출처를 명확히 제시하는 특화 RAG 시스템입니다.

## 1. 프로젝트 배경 및 목표
현재 재테크 정보는 네이버 블로그, 티스토리, 금융기관 공식 보고서 등 다양한 플랫폼에 파편화되어 있습니다. 신뢰할 수 있는 정보를 찾기 위해 여러 사이트를 직접 교차 검증해야 하며, 특히 금융기관의 공식 보고서는 PDF나 이미지 형태가 많아 일반적인 검색으로는 내용 확인이 어렵습니다.

이러한 문제를 해결하기 위해 텍스트, 이미지, 문서 등 다양한 포맷의 금융/재테크 정보를 하나의 벡터 공간에 통합하고, 사용자의 질문에 대해 명확한 출처와 함께 답변을 제공하는 RAG 기반 챗봇을 구현했습니다.

## 2. 주요 기능 및 기술적 구현

### 2.1. 다중 플랫폼 블로그 크롤러
각 플랫폼마다 다른 DOM 구조와 동적 로딩 방식을 하나로 통합 처리하는 크롤러를 구현했습니다.

*   **구현 내용:** 네이버, 티스토리, 브런치, 벨로그, 미디엄 등 5개 플랫폼 동시 지원
*   **이슈 해결:** 네이버 블로그의 경우 실제 콘텐츠가 iframe 내부에 숨겨져 있는 구조입니다. 메인 페이지에서 iframe의 `src` 속성을 추출한 뒤, 절대경로로 변환하여 내부 페이지를 재요청하는 방식으로 본문 텍스트를 확보했습니다.

```python
def extract_blog_content(url):
    domain = urlparse(url).netloc
    
    if "blog.naver.com" in domain:
        # iframe 구조 우회 처리
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        iframe = soup.find("iframe", id="mainFrame")
        
        if iframe and iframe.get("src"):
            iframe_url = urljoin(url, iframe["src"])
            response = requests.get(iframe_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find("div", class_="se-main-container")
            
    elif "tistory.com" in domain:
        content_div = soup.find("div", class_="tt_article_useless_p_margin")
    # ... 기타 플랫폼 처리 로직
```

### 2.2. 금융 문서 특화 OCR 파이프라인
금융 보고서에 포함된 표, 그래프, 수식을 일반 OCR로 처리할 때 발생하는 인식률 저하 문제를 개선하고자 했습니다.

*   **구현 내용:** Nanonets-OCR-s 멀티모달 모델을 활용하여 중소기업은행, 예금보험공사 등의 공식 보고서 처리
*   **구현 방식:** 이미지 크기 조절을 통한 메모리 최적화를 선행하고, 표는 HTML, 수식은 LaTeX 포맷으로 반환하도록 프롬프트를 설계하여 정형 데이터의 손실을 최소화했습니다.

### 2.3. RAG 파이프라인 구축
서로 다른 형식(블로그 글, OCR 텍스트, HWP 파일)의 데이터를 통일된 기준으로 벡터화했습니다.

*   **구현 내용:** FAISS 벡터 DB 구축 및 OpenAI 임베딩 활용
*   **최적화:** 재테크 정보의 문맥 유실을 막기 위해 `RecursiveCharacterTextSplitter`의 chunk_size를 1000, overlap을 200으로 설정하고, 의미적 검색 정확도를 높이기 위해 1024차원의 `text-embedding-3-small` 모델을 적용했습니다.

### 2.4. 웹 인터페이스 (Streamlit)
*   **구현 내용:** Streamlit을 활용한 사용자 친화적 챗봇 UI 구현
*   **상태 관리:** 세션 상태(session_state)를 활용해 대화 히스토리를 유지하고, 불필요한 대화 맥락을 사용자가 직접 지울 수 있도록 개별 삭제 기능을 추가했습니다.

## 3. 프로젝트 성과 및 데모

### 정량적 지표
*   **수집 데이터:** 5개 플랫폼 기반 70여 개 블로그 포스트 및 다수의 금융기관 보고서 처리
*   **응답 성능:** 검색 및 답변 생성까지 평균 2초 이내 응답 달성

### 챗봇 응답 예시
> **User:** "ISA에 대하여 알려줘"
> 
> **System:** "ISA(Individual Savings Account)는 개인종합자산관리계좌입니다. 연간 2,000만원까지 납입 가능하며, 5년간 세제혜택을 받을 수 있습니다. 주요 특징은 다음과 같습니다: 
> 1) 납입한도: 연간 2,000만원 
> 2) 의무가입기간: 3년 
> 3) 세제혜택: 투자수익 분리과세
> 
> *출처: blog_text33.txt - 'ISA 계좌 개설 방법 한도 만기'* "

## 4. 트러블슈팅 및 한계점

1. **대화 맥락 유실 문제**
   * **이슈:** `RetrievalQA` 체인 단독 사용 시 이전 대화를 기억하지 못하는 현상 발생.
   * **해결:** Streamlit의 `session_state`에 질의응답 히스토리를 배열 형태로 저장하고, 이를 UI에 렌더링함과 동시에 필요시 프롬프트에 맥락으로 전달할 수 있는 기반을 마련했습니다.

2. **실시간 정보 연동의 부재**
   * **이슈:** RAG 특성상 벡터 DB에 적재된 시점의 정보만 제공 가능하여, 매일 변동하는 예적금 금리나 환율 등 최신 트렌드 반영에 한계가 있었습니다.
   * **대응:** 현재 날짜 정보를 시스템 프롬프트에 주입하여 모델이 스스로 데이터의 시차를 인지하고 사용자에게 안내하도록 임시 조치했습니다. 향후 금융 API를 직접 연동하는 하이브리드 방식으로 개선할 예정입니다.

3. **이종 데이터 간 임베딩 품질 차이**
   * **이슈:** 구어체 위주의 블로그 글과 정형화된 공식 보고서 텍스트가 섞여 있어, 특정 질의에 대해 검색 품질이 편향되는 현상 발견.
   * **대응:** 메타데이터 필터링 구조를 추가하기 위한 사전 작업으로 소스별(source type) 태깅 처리를 진행했습니다.

## 5. 설치 및 실행 방법

**환경 요구사항**
* Python 3.9 이상
* OpenAI API Key

```bash
# 1. 저장소 클론
git clone https://github.com/sleepsoo/llm_chatbot_project_1.git
cd llm_chatbot_project_1

# 2. 의존성 패키지 설치
pip install streamlit langchain openai faiss-cpu beautifulsoup4 requests pillow transformers python-dotenv langchain-openai langchain-community

# 3. 환경 변수 설정 (.env 파일 생성)
echo "OPENAI_API_KEY=본인의_API_키" > .env

# 4. 앱 실행
streamlit run 6.final_app.py
```
*(참고: 데이터 파이프라인 구축 과정은 `1.crawling.ipynb` ~ `4.main.ipynb` 주피터 노트북을 순차적으로 실행하여 확인할 수 있습니다.)*

## 6. 프로젝트 구조
```text
llm_chatbot_project_1/
├── 1.crawling.ipynb          # 다중 플랫폼 크롤링 스크립트
├── 2.loader.ipynb            # 로컬 문서(HWP, TXT) 로드 모듈
├── 3.text_splitter.ipynb     # 데이터 텍스트 청킹 테스트
├── 4.main.ipynb              # OCR 처리 및 FAISS DB 구축
├── 6.final_app.py            # Streamlit 웹 애플리케이션
├── data/                     # 수집된 블로그, 금융 보고서, 이미지 원본
├── my_faiss_index/           # 생성된 벡터 데이터베이스
└── .env                      # 환경변수 (Git 제외)
```

## 7. 개발자 정보
* **GitHub:** [@sleepsoo](https://github.com/sleepsoo)
* 비전공자에서 AI 개발자로 커리어를 전환하며, 데이터 수집부터 LLM 서비스 서빙까지의 전체 파이프라인 구축을 목표로 진행한 개인 프로젝트입니다.

## APPENDIX

### 주요 라이브러리 선택 이유

| 기술 | 선택 이유 | 대안 대비 장점 |
|------|----------|---------------|
| **FAISS** | 대용량 벡터 검색 최적화 | Pinecone 대비 무료, ChromaDB 대비 속도 |
| **LangChain** | RAG 파이프라인 표준화 | 직접 구현 대비 안정성, 확장성 |
| **Nanonets-OCR-s** | 금융 문서 특화 시도 | Tesseract 대비 멀티모달 접근 |
| **Streamlit** | 빠른 프로토타이핑 | Flask 대비 개발속도, React 대비 단순성 |
| **BeautifulSoup** | 안정적인 HTML 파싱 | Selenium 대비 속도, Scrapy 대비 단순성 |

### 성능 최적화 기법
```python
# 1. 캐시 활용으로 리소스 절약
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
    return FAISS.load_local("my_faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2. 이미지 크기 최적화
resized_image = image.resize((image.width // 2, image.height // 2))

# 3. 효율적인 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 컨텍스트와 성능의 균형점
    chunk_overlap=200 # 의미 단절 방지
)
```

<p align="center">
  <strong>⭐ 이 프로젝트가 도움이 되셨다면 스타를 눌러주세요! ⭐</strong>
</p>

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=sleepsoo&repo=llm_chatbot_project_1&color=blueviolet"/>
</p>

---

*"좋은 기술은 복잡한 문제를 단순하게 해결합니다. 이 프로젝트가 재테크 정보 접근의 장벽을 낮추는 작은 기여가 되기를 바랍니다."*
