#!/usr/bin/env python3
import os
import json
import uuid
import serial           # pyserial for Bluetooth
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from modules.recoommender import recommend_fashion
from modules.weather_fetcher import prepare_arduino_weather_json

# Load environment variables
load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────
UPLOAD_FOLDER      = 'uploads'
PDF_PATH           = 'fashion_style_guide.pdf'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
BLUETOOTH_PORT     = os.getenv('BLUETOOTH_PORT', '/dev/cu.YEJIN')
BLUETOOTH_BAUD     = int(os.getenv('BLUETOOTH_BAUD', '9600'))

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """Check if the uploaded file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def send_via_bluetooth(payload: dict):
    """Send the JSON payload over Bluetooth serial to the Arduino."""
    try:
        ser = serial.Serial(BLUETOOTH_PORT, BLUETOOTH_BAUD, timeout=2)
        time.sleep(2)  # give Arduino time to reset and initialize
        ser.write((json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8'))
        ser.close()
    except Exception as e:
        app.logger.error(f"Bluetooth send failed: {e}")


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # 1) Save the uploaded image
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return "Invalid file", 400

        task_id = uuid.uuid4().hex
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(path)

        # 2) Run fashion recommendation and clean up Markdown fences
        rec = recommend_fashion(path, PDF_PATH)
        if isinstance(rec, str):
            s = rec.strip()
            if s.startswith('```'):
                lines = s.splitlines()
                if lines and lines[0].startswith('```'):
                    lines.pop(0)
                if lines and lines[-1].startswith('```'):
                    lines.pop(-1)
                s = '\n'.join(lines)
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                rec = { 'error': s }

        # 3) Run weather JSON generator
        w_out = prepare_arduino_weather_json()
        if isinstance(w_out, str):
            try:
                weather = json.loads(w_out)
            except json.JSONDecodeError:
                weather = {}
        else:
            weather = w_out

        # 4) Build payload: message or reason, color as top/bottom, weather with min/max temps
        payload = {
            'message':    rec.get('suggestion') or rec.get('reason') or "",
            'color':      f"{rec.get('top','')}/{rec.get('bottom','')}",
            'weather': {
                '날씨':     weather.get('날씨'),
                '최저기온': weather.get('최저기온'),
                '최고기온': weather.get('최고기온'),
            }
        }

        # 5) Send to Arduino via Bluetooth
        send_via_bluetooth(payload)

        return jsonify({ 'status': 'sent', 'payload': payload })

    # GET: show upload form
    return '''
      <h2>Fashion Upload & Push to Arduino</h2>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <button type="submit">Upload & Send</button>
      </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
