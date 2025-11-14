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
from model import My_Advanced_Model

#  FASTAPI INIT 
app = FastAPI(title="Emotion Detection")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

#  DEVICE 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LOAD MODELS 
# Emotion model
emotion_model = My_Advanced_Model(5)
emotion_model.load_state_dict(torch.load("emotion_model_21_02.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()

# YOLO model
yolo_model = YOLO("model_yolo.pt")
class_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transform ảnh
transform = transforms.Compose([
    transforms.ToTensor(),
])

# WEBCAM STREAM GENERATOR
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Dò mặt bằng YOLO
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
                conf = conf.item() * 100
                label = class_labels[pred.item()]

            text = f"{label} ({conf:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # Encode frame để stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()

        pred = np.argmax(probs)
        label = class_labels[pred]
        conf = probs[pred] * 100

        # Vẽ lên ảnh
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Lưu output của từng người
        faces_output.append({
            "id": face_id,
            "label": label,
            "confidence": round(conf, 2),
            "probabilities": {class_labels[i]: float(f"{probs[i]*100:.2f}") for i in range(len(class_labels))}
        })

        face_id += 1

    # Encode ảnh
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "image_base64": img_base64,
        "faces": faces_output
    })
