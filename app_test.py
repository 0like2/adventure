from flask import Flask, jsonify
import serial
import time

app = Flask(__name__)

# 송신(Outgoing) COM 포트 사용
BLUETOOTH_PORT = '/dev/cu.YEJIN'  # 실제 송신 포트 번호로 변경
BLUETOOTH_RATE = 9600


@app.route('/send_data')
def send_data():
    try:
        # 블루투스 시리얼 연결
        ser = serial.Serial(BLUETOOTH_PORT, BLUETOOTH_RATE, timeout=1)
        time.sleep(2)  # 연결 안정화 대기

        # 데이터 전송
        data = '{"message": "Hello LCD", "value": 123}'
        ser.write(data.encode())

        # 응답 확인 (선택사항)
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        if response:
            print(f"응답 수신: {response}")

        ser.close()
        return jsonify({"status": "success", "message": "데이터가 전송되었습니다."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
