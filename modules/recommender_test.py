import openai
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from weather_fetcher import get_weather_info

# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def direct_rgb(mode="white_navy"):
    if mode == "white_navy":
        return [245, 245, 245], [0, 40, 85]  # 화이트 셔츠 + 네이비 슬랙스
    elif mode == "beige_black":
        return [220, 200, 180], [20, 20, 20]  # 베이지 니트 + 블랙 팬츠
    elif mode == "gray_charcoal":
        return [180, 180, 180], [50, 50, 50]  # 라이트그레이 + 차콜
    elif mode == "khaki_white":
        return [100, 120, 80], [250, 250, 250]  # 카키 + 화이트
    elif mode == "navy_burgundy":
        return [10, 20, 70], [128, 0, 32]  # 네이비 + 버건디
    elif mode == "black_ivory":
        return [10, 10, 10], [240, 240, 230]  # 블랙 + 아이보리
    else:
        return [245, 245, 245], [0, 40, 85]  # 기본값

def direct_weather(mode="rain"):
    """
    날씨 키워드를 반환합니다.
    """
    weather_data = {
        "rain": {
            "weather": "비"
        },
        "sunny": {
            "weather": "맑음"
        },
        "cloudy": {
            "weather": "흐림"
        },
        "snow": {
            "weather": "눈"
        },
        "default": {
            "weather": "맑음"
        }
    }

    return weather_data.get(mode, weather_data["default"])

def build_vector_store(pdf_path, persist_directory="db"):
    """
    PDF 파일을 읽어 벡터스토어로 저장합니다.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)

    vector_store.persist()
    return vector_store

def search_context(query, persist_directory="db", top_k=3):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    docs = vector_store.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

def make_prompt(top_rgb, bottom_rgb, context, weather_text="맑음"):
    prompt = f"""
당신은 전문 패션 스타일리스트입니다.

오늘의 날씨는 '{weather_text}'입니다. 날씨와 의상 색상 조합을 모두 고려하여 코디의 적절성을 평가해주세요.

아래는 참고할 스타일 가이드입니다:
------------------------------
{context}
------------------------------

지금 사용자가 입은 옷의 색상은 다음과 같습니다.
- 상의 RGB: {tuple(top_rgb)}
- 하의 RGB: {tuple(bottom_rgb)}

질문:
1. 이 옷 색상 조합은 '{weather_text}' 날씨에 어울리며 세련된 조합인가요? 전반적으로 잘 입은 편인가요?
2. 어울리지 않거나 개선이 필요하다면, 어떤 색상 또는 스타일로 조정하는 것이 좋을까요?
3. 참고한 스타일 가이드를 기반으로, 날씨와 상황에 맞는 더 고급스럽고 실용적인 코디를 구체적으로 추천해주세요. (예시 색상, 톤, 아이템을 포함하여 설명)

답변은 구체적으로 해주세요. (예시 색깔이나 톤을 제시)
"""
    return prompt


# 날씨 축가 : rain / sunny 등을 받으면 gpt 돌리지 않고 그냥 결과 몇가지 나누어서 텍스트를 붙여서 출력해야함
def recommend_fashion(image_path, pdf_path, mode="white_navy", weather_mode="sunny"):
    # Step 1. 색상 설정
    top_rgb, bottom_rgb = direct_rgb(mode)

    # Step 2. 날씨 정보 추출
    weather_data = get_weather_info()
    weather_text = weather_data["날씨"]

    '''
    weather_info = direct_weather(weather_mode)
    weather_text = weather_info["weather"]
    '''

    # Step 3. 벡터스토어
    if not os.path.exists("../db"):
        print("벡터스토어를 새로 생성합니다...")
        build_vector_store(pdf_path)

    # Step 4. 스타일 context 검색
    query = f"상의 RGB {tuple(top_rgb)}, 하의 RGB {tuple(bottom_rgb)}에 어울리는 스타일 추천"
    context = search_context(query)

    # Step 5. 프롬프트 생성
    prompt = make_prompt(top_rgb, bottom_rgb, context, weather_text)

    # Step 6. GPT 호출
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    image_path = "../images.jpeg"
    pdf_path = "../fashion_style_guide.pdf"

    # 예시: 맑은 날
    result = recommend_fashion(image_path, pdf_path, mode="white_navy", weather_mode="sunny")
    print("\n[패션 추천 결과]\n")
    print(result)
