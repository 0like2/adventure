#!/usr/bin/env python3
# main.py

import argparse
import os
from dotenv import load_dotenv
from modules.recoommender import recommend_fashion
from modules.weather_fetcher import prepare_arduino_weather_json
from modules.color_extractor import process_clothing

# 1) .env에서 키 로드
load_dotenv()

# 2) Google Vision 크레덴셜 설정 (선택)
google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_creds:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", google_creds)


def main():
    parser = argparse.ArgumentParser(
        description="👗 패션 추천 + 🌤️ 아두이노용 날씨 JSON 생성 스크립트"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="입력 이미지 파일 경로"
    )
    args = parser.parse_args()

    image_path = args.image
    pdf_path = "fashion_style_guide.pdf"  # 고정 경로
    output_weather_json = ""  # 저장 경로(비어 있으면 저장 안 함)

    # ─── 1) 패션 추천 ─────────────────────────────────────────────────────────
    print("📦 패션 추천을 수행합니다...")
    try:
        recommendation = recommend_fashion(image_path, pdf_path)
        print("\n[패션 추천 결과]\n")
        print(recommendation)
    except Exception as err:
        print(f"❌ 패션 추천 중 오류 발생: {err}")

    # ─── 2) 아두이노용 날씨 JSON 생성 ───────────────────────────────────────────
    print("\n📡 아두이노용 날씨 JSON을 생성합니다...")
    try:
        weather_json = prepare_arduino_weather_json()
        print("\n[아두이노용 날씨 JSON]\n")
        print(weather_json)

        if output_weather_json:
            with open(output_weather_json, "w", encoding="utf-8") as f:
                f.write(weather_json)
            print(f"\n✅ JSON 파일로 저장됨: {output_weather_json}")
    except Exception as err:
        print(f"❌ 날씨 정보 생성 중 오류 발생: {err}")


if __name__ == "__main__":
    main()
