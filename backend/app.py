from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from io import BytesIO
import base64
from backend.model import My_Advanced_Model
import os

# PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "static")

# FASTAPI INIT
app = FastAPI(title="Emotion Detection")
templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Emotion model
emotion_model = My_Advanced_Model(5)
emotion_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "emotion_model_21_02.pth"),
    map_location=device
))
emotion_model.to(device)
emotion_model.eval()

# YOLO model
yolo_model = YOLO(os.path.join(BASE_DIR, "model_yolo.pt"))
class_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# WEBCAM STREAM
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            roi = transform(gray).unsqueeze(0).to(device)

            with torch.no_grad():
                output = emotion_model(roi)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
                label = class_labels[pred.item()]
                conf = conf.item() * 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.1f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

# ROUTES
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = yolo_model(frame, verbose=False)[0]

    faces_output = []
    face_id = 1

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        roi = transform(gray).unsqueeze(0).to(device)

        with torch.no_grad():
            output = emotion_model(roi)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy().tolist()  # <-- chuyển sang list

        pred = int(np.argmax(probs))
        label = str(class_labels[pred])
        conf = float(probs[pred] * 100)

        # Vẽ bounding box và nhãn
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        faces_output.append({
            "id": int(face_id),
            "label": label,
            "confidence": round(conf, 2),
            "probabilities": {class_labels[i]: float(probs[i]*100) for i in range(len(class_labels))}
        })
        face_id += 1

    # Encode ảnh kết quả sang base64
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "image_base64": img_base64,
        "faces": faces_output
    })
