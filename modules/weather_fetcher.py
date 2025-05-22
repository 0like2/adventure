import requests
import geocoder
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather_info():
    g = geocoder.ip('me')
    lat, lon = g.latlng

    # 현재 날씨 + 예보 정보
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=kr"
    resp = requests.get(url)
    data = resp.json()

    # 오늘 날짜 기준으로 필터
    today = datetime.now().strftime('%Y-%m-%d')
    today_data = [item for item in data['list'] if item['dt_txt'].startswith(today)]

    temps = [entry['main']['temp'] for entry in today_data]
    min_temp = min(temps)
    max_temp = max(temps)
    weather_description = today_data[0]['weather'][0]['description']

    return {
        "최저기온": round(min_temp, 1),
        "최고기온": round(max_temp, 1),
        "날씨": weather_description
    }

def prepare_arduino_weather_json():
    weather_data = get_weather_info()
    return json.dumps(weather_data, ensure_ascii=False)  # {'날씨': '맑음', '최저기온': 14.3, '최고기온': 21.7}


if __name__ == "__main__":
    print("📡 오늘의 날씨 정보를 가져오는 중...\n")

    weather_data = get_weather_info()
    print("✅ 날씨 정보:")
    for k, v in weather_data.items():
        print(f"- {k}: {v}")

    print("\n📦 아두이노용 JSON:")
    print(prepare_arduino_weather_json())

