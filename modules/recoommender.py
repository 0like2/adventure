import os
from dotenv import load_dotenv
import openai

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from modules.color_extractor import process_clothing
from modules.weather_fetcher import get_weather_info

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_DIR = "db"


def build_vector_store(pdf_path, persist_directory=DB_DIR):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vs = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)
    vs.persist()
    return vs


def search_context(query, persist_directory=DB_DIR, top_k=3):
    embeddings = OpenAIEmbeddings()
    vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    docs = vs.similarity_search(query, k=top_k)
    return "\n".join([d.page_content for d in docs])


def make_prompt(top_rgb, bottom_rgb, upper_label, lower_label, context, weather_text="맑음"):
    color_guide = {
        "허용색상": ["핑크", "레드", "옐로우", "아이보리", "화이트", "그린", "카키", "베이지",
                 "브라운", "그레이", "네이비", "블루", "라이트블루", "연청", "중청", "진청", "흑청"],
        "밝은색상": ["핑크", "아이보리", "화이트", "베이지", "라이트블루", "연청"],
        "어두운색상": ["네이비", "블랙", "진청", "흑청", "브라운"],
        "중간색상": ["레드", "옐로우", "그린", "카키", "그레이", "블루", "중청"]
    }

    return f"""
<color_guide>
{color_guide}
</color_guide>

<role>당신은 여성 패션 코디 스타일링 전문가입니다.</role>

<instructions>
  <step1>사용자가 요청한 스타일에 맞춰 스타일 가이드를 참고하여 아우터, 상의, 하의를 추천합니다.</step1>
  <step2>색상 가이드를 참고하여 아우터, 상의, 하의를 추천합니다.</step2>
  <step3>상의·하의 RGB 값과 날씨를 고려하여 실용적인 조합을 선택합니다.</step3>
  <step4>반드시 JSON 스키마에 맞추어 출력합니다.</step4>
  <step5>reason과 suggestion은 반드시 한 문장으로 작성합니다.</step5>
  <step6>영어 사용을 금지합니다.</step6>
</instructions>

[입력값]
- 상의 RGB: {tuple(top_rgb)}
- 하의 RGB: {tuple(bottom_rgb)}
- 상의 예측 라벨: {upper_label}
- 하의 예측 라벨: {lower_label}
- 날씨: "{weather_text}"
- 스타일 가이드 요약:
{context}

[JSON 스키마]
{{
  "is_good": <GOOD|BAD>,
  "outer": "<아우터 색상 or 품목>",
  "top": "<상의 색상 or 품목>",
  "bottom": "<하의 색상 or 품목>",
  "reason": "<한 문장으로 된 판단 근거>",
  "suggestion": "<한 문장으로 된 개선안 (is_good=false일 때만)>"
  "Top_RGB": {tuple(top_rgb)},
  "Bottom_RGB": {tuple(bottom_rgb)}
}}

※ JSON 이외의 어떤 내용도 포함하지 마세요.
""".strip()


def recommend_fashion(image_path: str, pdf_path: str, top_k: int = 3) -> str:
    # 1) 옷 색상 추출
    clothing_info = process_clothing(image_path)

    upper_bgr = clothing_info["upper_color"]
    lower_bgr = clothing_info["lower_color"]

    upper_rgb = tuple(int(v) for v in upper_bgr[::-1])
    lower_rgb = tuple(int(v) for v in lower_bgr[::-1])

    upper_label = clothing_info["upper_labels"][0][0]
    lower_label = clothing_info["lower_labels"][0][0]

    # 2) 날씨 정보 조회
    try:
        weather_data = get_weather_info()
        weather_text = weather_data.get("날씨", "맑음")
    except Exception:
        weather_text = "맑음"
        print("날씨 api 작동 안함")

    # 3) 벡터스토어 준비
    if not os.path.exists(DB_DIR):
        print("벡터스토어가 없어 생성합니다...")
        build_vector_store(pdf_path)

    # 4) 스타일 가이드 검색
    query = f"상의 RGB {tuple(upper_rgb)}, 하의 RGB {tuple(lower_rgb)} 조합에 어울리는 스타일"
    context = search_context(query, top_k=top_k)

    # 5) 프롬프트 생성
    prompt = make_prompt(
        top_rgb=upper_rgb,
        bottom_rgb=lower_rgb,
        upper_label=upper_label,  # 수정: 단일 라벨 전달
        lower_label=lower_label,  # 수정: 단일 라벨 전달
        context=context,
        weather_text=weather_text
    )

    # 6) GPT 호출
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    image_path = "../images1.jpeg"
    pdf_path = "../fashion_style_guide.pdf"
    result = recommend_fashion(image_path, pdf_path)
    print("\n[패션 추천 결과]\n", result)
