아래는 README.md에서 보기 좋게 구조화된 프로젝트 디렉토리 설명입니다. 각 폴더와 파일에 대한 설명도 마크다운 형식으로 정리했습니다. 그대로 복붙해서 사용하시면 됩니다:

⸻


# 👕 Fashion Recommendation Project

이 프로젝트는 아두이노와 연동하여 **실시간 패션 이미지 분석** 및 **의상 추천** 기능을 제공하는 시스템입니다.  
아두이노 버튼 클릭 → 이미지 캡처 → 분석 → 추천 결과를 다시 아두이노로 전송하는 전체 파이프라인을 구성합니다.

---

## 📁 프로젝트 디렉토리 구조

fashion_recommendation_project/
├── main.py                     # 전체 파이프라인 제어 (입력 → 분석 → 출력)
├── config.py                   # 환경 설정 (경로, 포트 등)
├── requirements.txt            # Python 의존성 목록
│
├── utils/                      # 공통 유틸리티 함수 모음
│   ├── serial_utils.py         # 아두이노 시리얼 통신 함수
│   └── image_utils.py          # 이미지 저장 및 전처리
│
├── modules/                    # 주요 기능 모듈
│   ├── cloth_detector.py       # 상하의 객체 감지 (YOLO or Vision API)
│   ├── color_extractor.py      # 색상 추출 (KMeans 등)
│   ├── recommender.py          # 의상 추천 로직 (LLM 기반 또는 규칙 기반)
│   └── weather_fetcher.py      # 날씨 정보 수집 (외부 API 활용)
│
└── data/                       # 데이터 저장소
    ├── captured/               # 저장된 이미지
    └── result/                 # 분석 결과 (JSON, 텍스트 등)

## 🛠️ 주요 기능 요약

| 기능 | 설명 |
|------|------|
| 🎯 전체 제어 | `main.py`에서 입력 → 분석 → 결과 전송의 모든 흐름을 제어 |
| 📸 이미지 처리 | 아두이노로부터 이미지 저장 후, 전처리 수행 |
| 🧥 옷 감지 | 상의/하의 객체 감지를 통해 색상 영역 분리 |
| 🎨 색상 추출 | 감지된 영역에서 주요 색상 추출 (KMeans 기반) |
| 🌤️ 날씨 API | 현재 날씨에 따라 추천 결과 반영 |
| 💡 추천 시스템 | 상의/하의 색상 + 날씨를 기반으로 의상 추천 |
| 🔁 아두이노 통신 | 버튼 → 이미지 저장, 분석 후 LED 신호 전송 등 |

---

## 🔧 실행 예시

```bash
python main.py


⸻

📦 설치

pip install -r requirements.txt


⸻

🔐 환경 변수 (.env 예시)

OPENAI_API_KEY=your_key_here
SERIAL_PORT=/dev/ttyUSB0
BAUDRATE=9600
WEATHER_API_KEY=your_weather_api_key


⸻

📬 문의

본 프로젝트에 관한 문의는 이영락에게 주세요.

---

필요하면 `main.py`의 플로우 차트나 예제 입출력도 추가해드릴 수 있어요. 시각화 다이어그램도 넣을까요?
