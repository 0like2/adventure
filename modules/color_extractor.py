import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from google.cloud import vision
import io
import os
from dotenv import load_dotenv

load_dotenv()
# 환경 변수로 크레덴셜 설정
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "adventure-459704-b405114d6880.json")

# Vision API 클라이언트 초기화
client = vision.ImageAnnotatorClient()

# ========== 초기화 ==========
mp_pose = mp.solutions.pose
mp_seg = mp.solutions.selfie_segmentation
pose = mp_pose.Pose(static_image_mode=True)
segmenter = mp_seg.SelfieSegmentation(model_selection=1)


# ========== 화이트 밸런스 ==========
def white_balance(img: np.ndarray) -> np.ndarray:
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(img)


# ========== LAB 기반 dominant color 추출 ==========
def extract_dominant_color_lab(region: np.ndarray, mask_region: np.ndarray, k: int = 3) -> np.ndarray:
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3)
    mask_flat = mask_region.reshape(-1)
    valid_pixels = pixels[mask_flat > 0]
    valid_pixels = np.array([p for p in valid_pixels if not np.all(p == 0)])
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(valid_pixels)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[np.argmax(counts)]
    lab_color = np.uint8([[dominant]])
    bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0][0]
    return bgr_color


# ========== 팔레트 이미지 생성 ==========
def draw_palette(color_upper: np.ndarray, color_lower: np.ndarray) -> np.ndarray:
    palette = np.zeros((100, 200, 3), dtype=np.uint8)
    palette[:, :100] = color_upper
    palette[:, 100:] = color_lower
    return palette


# ========== 박스 계산 ==========
def get_body_boxes(landmarks, h: int, w: int):
    top = int(min(landmarks[11].y, landmarks[12].y) * h)
    mid = int((landmarks[23].y + landmarks[24].y) / 2 * h)
    bottom = int((landmarks[25].y + landmarks[26].y) / 2 * h)
    return (0, top, w, mid), (0, mid, w, bottom)


# ========== 마스크 적용하고 크롭 ==========
def apply_mask_and_crop(frame: np.ndarray, mask: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask_uint = (mask * 255).astype(np.uint8)
    masked = cv2.bitwise_and(frame, frame, mask=mask_uint)
    return masked[y1:y2, x1:x2]


# ========== Cloud Vision API 라벨 탐지 ==========
def detect_labels(image_bytes: bytes, max_results: int = 10) -> list:
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image)
    return [(lbl.description, lbl.score) for lbl in response.label_annotations[:max_results]]


# ========== 전체 처리 함수 ==========
def process_clothing(image_path: str, apply_wb: bool = True) -> dict:
    frame = cv2.imread(image_path)
    if apply_wb:
        frame = white_balance(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Segmentation
    mask = segmenter.process(rgb).segmentation_mask > 0.5

    # Pose detection
    pose_result = pose.process(rgb)
    if not pose_result.pose_landmarks:
        raise RuntimeError("사람을 인식하지 못했습니다.")
    landmarks = pose_result.pose_landmarks.landmark

    h, w, _ = frame.shape
    upper_box, lower_box = get_body_boxes(landmarks, h, w)
    upper_region = apply_mask_and_crop(frame, mask, upper_box)
    lower_region = apply_mask_and_crop(frame, mask, lower_box)
    upper_mask_crop = mask[upper_box[1]:upper_box[3], upper_box[0]:upper_box[2]]
    lower_mask_crop = mask[lower_box[1]:lower_box[3], lower_box[0]:lower_box[2]]

    upper_color = extract_dominant_color_lab(upper_region, upper_mask_crop)
    lower_color = extract_dominant_color_lab(lower_region, lower_mask_crop)


    x1, y1, x2, y2 = upper_box
    upper_crop = frame[y1:y2, x1:x2]
    success, upper_buf = cv2.imencode('.jpg', upper_crop)
    if not success:
        raise RuntimeError("upper_crop 인코딩 실패")
    upper_labels = detect_labels(upper_buf.tobytes())

    x1, y1, x2, y2 = lower_box
    lower_crop = frame[y1:y2, x1:x2]
    success, lower_buf = cv2.imencode('.jpg', lower_crop)
    if not success:
        raise RuntimeError("lower_crop 인코딩 실패")
    lower_labels = detect_labels(lower_buf.tobytes())

    palette = draw_palette(upper_color, lower_color)
    return {
        'upper_color': upper_color,
        'lower_color': lower_color,
        'upper_labels': upper_labels,
        'lower_labels': lower_labels,
        'palette': palette,
        'upper_region': upper_region,
        'lower_region': lower_region
    }


# ========== 스크립트 직접 실행 ==========
if __name__ == '__main__':
    sample_image = '../images1.jpeg'
    result = process_clothing(sample_image)

    # ─── 결과 출력 ─────────────────────────────────────────
    # 상의 dominant color (BGR) 및 RGB
    upper_bgr = result['upper_color']
    upper_rgb = tuple(int(v) for v in upper_bgr[::-1])
    print("=== 상의 분석 결과 ===")
    print(f"Dominant Color (BGR): {upper_bgr}")
    print(f"Dominant Color (RGB): {upper_rgb}")
    print("예측 라벨:")
    for desc, score in result['upper_labels']:
        print(f" - {desc} ({score:.2f})")
    print()

    # 하의 dominant color (BGR) 및 RGB
    lower_bgr = result['lower_color']
    lower_rgb = tuple(int(v) for v in lower_bgr[::-1])
    print("=== 하의 분석 결과 ===")
    print(f"Dominant Color (BGR): {lower_bgr}")
    print(f"Dominant Color (RGB): {lower_rgb}")
    print("예측 라벨:")
    for desc, score in result['lower_labels']:
        print(f" - {desc} ({score:.2f})")
    print()

    # ─── 저장 부분 ─────────────────────────────────────────
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(sample_image))[0]
    palette_path = os.path.join(output_dir, f"{base_name}_palette.jpg")
    cv2.imwrite(palette_path, result['palette'])
    upper_path   = os.path.join(output_dir, f"{base_name}_upper.jpg")
    cv2.imwrite(upper_path, result['upper_region'])
    lower_path   = os.path.join(output_dir, f"{base_name}_lower.jpg")
    cv2.imwrite(lower_path, result['lower_region'])

    print(f" 결과 저장 완료:\n - {palette_path}\n - {upper_path}\n - {lower_path}")

