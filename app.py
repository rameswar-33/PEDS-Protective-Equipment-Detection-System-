from pathlib import Path
import io
import os
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
# The supplied PPE model remains an independent detector.
PPE_MODEL_PATH = BASE_DIR / os.getenv("PPE_MODEL", "models/ppe_best.onnx")
PPE_CONFIDENCE = float(os.getenv("PPE_CONF", "0.35"))
PPE_IOU = float(os.getenv("PPE_IOU", "0.45"))
PPE_IMG_SIZE = int(os.getenv("PPE_IMGSZ", "640"))

# The second detector is a two-class, YOLO-compatible model converted to
# ONNX for portable, fully local inference with ONNX Runtime.
FIRE_MODEL_PATH = BASE_DIR / os.getenv(
    "FIRE_MODEL", "models/fire_smoke_best.onnx"
)
FIRE_CONFIDENCE = float(os.getenv("FIRE_CONF", "0.40"))
FIRE_IOU = float(os.getenv("FIRE_IOU", "0.45"))
FIRE_IMG_SIZE = 640
FIRE_CLASSES = ("fire", "smoke")

# The supplied model's embedded metadata contains these classes.
# Keep the exact model order; do not reorder them.
PPE_MODEL_CLASSES = [
    "helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots"
]

# Classes displayed as the currently implemented PPE checks.
TRACKED_CLASSES = {"helmet", "gloves", "vest", "boots"}
MISSING_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}

if not PPE_MODEL_PATH.exists():
    raise RuntimeError(f"PPE model not found: {PPE_MODEL_PATH}")
if not FIRE_MODEL_PATH.exists():
    raise RuntimeError(f"Fire/smoke model not found: {FIRE_MODEL_PATH}")

ppe_model = YOLO(str(PPE_MODEL_PATH), task="detect")


class FireSmokeDetector:
    """Small local ONNX Runtime runner for the two-class YOLO model."""

    def __init__(self, model_path: Path, confidence: float, iou: float) -> None:
        self.confidence = confidence
        self.iou = iou
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])
        if (self.input_width, self.input_height) != (FIRE_IMG_SIZE, FIRE_IMG_SIZE):
            raise RuntimeError(
                "Fire/smoke model must use a 640x640 input; "
                f"received {self.input_width}x{self.input_height}"
            )

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            self.input_height - resized_height - pad_y,
            pad_x,
            self.input_width - resized_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, pad_x, pad_y

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        padded, scale, pad_x, pad_y = self._letterbox(image)
        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0
        predictions = np.squeeze(
            self.session.run(None, {self.input_name: tensor})[0], axis=0
        )

        # Model output: cx, cy, width, height, objectness, fire, smoke.
        class_scores = predictions[:, 5 : 5 + len(FIRE_CLASSES)]
        class_ids = class_scores.argmax(axis=1)
        confidences = predictions[:, 4] * class_scores[np.arange(len(predictions)), class_ids]
        candidates = np.where(confidences >= self.confidence)[0]
        height, width = image.shape[:2]
        boxes_by_class: dict[int, list[list[int]]] = {idx: [] for idx in range(len(FIRE_CLASSES))}
        scores_by_class: dict[int, list[float]] = {idx: [] for idx in range(len(FIRE_CLASSES))}

        for index in candidates:
            center_x, center_y, box_width, box_height = predictions[index, :4]
            x1 = max(0.0, (center_x - box_width / 2 - pad_x) / scale)
            y1 = max(0.0, (center_y - box_height / 2 - pad_y) / scale)
            x2 = min(float(width), (center_x + box_width / 2 - pad_x) / scale)
            y2 = min(float(height), (center_y + box_height / 2 - pad_y) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[index])
            boxes_by_class[class_id].append(
                [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]
            )
            scores_by_class[class_id].append(float(confidences[index]))

        detections: list[dict[str, Any]] = []
        for class_id, boxes in boxes_by_class.items():
            if not boxes:
                continue
            kept_indices = cv2.dnn.NMSBoxes(
                boxes, scores_by_class[class_id], score_threshold=0.0, nms_threshold=self.iou
            )
            for kept_index in np.asarray(kept_indices).reshape(-1):
                x, y, box_width, box_height = boxes[int(kept_index)]
                detections.append(
                    {
                        "class_id": class_id,
                        "class": FIRE_CLASSES[class_id],
                        "confidence": scores_by_class[class_id][int(kept_index)],
                        "bbox": [float(x), float(y), float(x + box_width), float(y + box_height)],
                        "detector": "fire_smoke",
                    }
                )
        return sorted(detections, key=lambda detection: detection["confidence"], reverse=True)


fire_smoke_model = FireSmokeDetector(FIRE_MODEL_PATH, FIRE_CONFIDENCE, FIRE_IOU)

app = FastAPI(title="Local PPE & Fire/Smoke Detection System", version="2.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def _class_name(names: Any, idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(idx, PPE_MODEL_CLASSES[idx] if idx < len(PPE_MODEL_CLASSES) else idx))
    if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
        return str(names[idx])
    return PPE_MODEL_CLASSES[idx] if 0 <= idx < len(PPE_MODEL_CLASSES) else str(idx)


def detect_image(image: np.ndarray) -> dict:
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    result = ppe_model.predict(
        source=image,
        imgsz=PPE_IMG_SIZE,
        conf=PPE_CONFIDENCE,
        iou=PPE_IOU,
        verbose=False,
        device="cpu",
    )[0]

    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", None) or getattr(ppe_model, "names", None)

        h, w = image.shape[:2]
        for box, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = _class_name(names, int(cls_id))
            detections.append({
                "class_id": int(cls_id),
                "class": label,
                "confidence": float(conf),
                "bbox": [
                    max(0, min(w, x1)), max(0, min(h, y1)),
                    max(0, min(w, x2)), max(0, min(h, y2)),
                ],
                "detector": "ppe",
            })

    fire_smoke_detections = fire_smoke_model.predict(image)
    detections.extend(fire_smoke_detections)

    counts = {name: 0 for name in TRACKED_CLASSES}
    missing = {name: 0 for name in MISSING_CLASSES}
    fire_smoke_counts = {name: 0 for name in FIRE_CLASSES}
    for d in detections:
        if d["class"] in counts:
            counts[d["class"]] += 1
        if d["class"] in missing:
            missing[d["class"]] += 1
        if d["class"] in fire_smoke_counts:
            fire_smoke_counts[d["class"]] += 1

    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "detections": detections,
        "counts": counts,
        "missing_counts": missing,
        "fire_smoke_counts": fire_smoke_counts,
        "fire_smoke_detections": fire_smoke_detections,
        "tracked_classes": sorted(TRACKED_CLASSES),
        "model_classes": PPE_MODEL_CLASSES,
        "fire_smoke_classes": list(FIRE_CLASSES),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": PPE_MODEL_PATH.name,
        "models": {"ppe": PPE_MODEL_PATH.name, "fire_smoke": FIRE_MODEL_PATH.name},
        "backends": {"ppe": "Ultralytics + ONNX Runtime", "fire_smoke": "ONNX Runtime"},
        "offline": True,
        "confidence": {"ppe": PPE_CONFIDENCE, "fire_smoke": FIRE_CONFIDENCE},
        "iou": {"ppe": PPE_IOU, "fire_smoke": FIRE_IOU},
        "imgsz": {"ppe": PPE_IMG_SIZE, "fire_smoke": FIRE_IMG_SIZE},
        "classes": {"ppe": PPE_MODEL_CLASSES, "fire_smoke": list(FIRE_CLASSES)},
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    try:
        return JSONResponse(detect_image(image))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/detect-frame")
async def detect_frame(file: UploadFile = File(...)):
    return await detect(file)


@app.post("/api/annotated")
async def annotated(file: UploadFile = File(...)):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        plotted = image.copy()
        colors = {"fire": (40, 70, 245), "smoke": (190, 190, 190)}
        for detection in detect_image(image)["detections"]:
            x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox"])
            label = detection["class"]
            color = colors.get(label, (214, 214, 214))
            cv2.rectangle(plotted, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {detection['confidence']:.0%}"
            cv2.putText(
                plotted, text, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    ok, encoded = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result")
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")

# --- Notification Configs & Models ---

telegram_config = {
    "bot_token": "",
    "chat_id": "",
    "configured": False
}

class TelegramConfig(BaseModel):
    bot_token: str
    chat_id: str

class TelegramMessage(BaseModel):
    message: str

whatsapp_config = {
    "access_token": "",
    "phone_number_id": "",
    "recipient": "",
    "template_name": "",
    "template_language": "en_US",
    "token_hint": "",
    "configured": False
}

class WhatsAppConfig(BaseModel):
    access_token: str = ""
    phone_number_id: str = ""
    recipient: str = ""
    template_name: str = ""
    template_language: str = "en_US"

class WhatsAppMessage(BaseModel):
    message: str
    params: list[str] = []

# --- Telegram Endpoints ---

@app.get("/api/telegram/config")
def get_telegram_config():
    return {
        "bot_token": telegram_config["bot_token"],
        "chat_id": telegram_config["chat_id"],
        "configured": telegram_config["configured"]
    }

@app.post("/api/telegram/config")
def set_telegram_config(config: TelegramConfig):
    telegram_config["bot_token"] = config.bot_token
    telegram_config["chat_id"] = config.chat_id
    telegram_config["configured"] = bool(config.bot_token and config.chat_id)
    return {"ok": True, "configured": telegram_config["configured"]}

@app.post("/api/telegram/test")
async def test_telegram():
    if not telegram_config["configured"]:
        raise HTTPException(status_code=400, detail="Telegram not configured")
    url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
    payload = {
        "chat_id": telegram_config["chat_id"],
        "text": "<b>Test Message</b>\nThis is a test notification from the PPE system.",
        "parse_mode": "HTML"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return {"ok": True, "details": resp.json()}

@app.post("/api/telegram/notify")
async def notify_telegram(msg: TelegramMessage):
    if not telegram_config["configured"]:
        raise HTTPException(status_code=400, detail="Telegram not configured")
    url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
    payload = {
        "chat_id": telegram_config["chat_id"],
        "text": msg.message,
        "parse_mode": "HTML"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return {"ok": True}


# --- WhatsApp Endpoints ---

@app.get("/api/whatsapp/config")
def get_whatsapp_config():
    return {
        "phone_number_id": whatsapp_config["phone_number_id"],
        "recipient": whatsapp_config["recipient"],
        "template_name": whatsapp_config["template_name"],
        "template_language": whatsapp_config["template_language"],
        "token_hint": whatsapp_config["token_hint"],
        "configured": whatsapp_config["configured"]
    }

@app.post("/api/whatsapp/config")
def set_whatsapp_config(config: WhatsAppConfig):
    # Only update access_token if a new one was provided (non-empty)
    if config.access_token:
        whatsapp_config["access_token"] = config.access_token
        token_len = len(config.access_token)
        if token_len > 8:
            whatsapp_config["token_hint"] = "..." + config.access_token[-8:]
        else:
            whatsapp_config["token_hint"] = config.access_token

    if config.phone_number_id:
        whatsapp_config["phone_number_id"] = config.phone_number_id
    if config.recipient:
        whatsapp_config["recipient"] = config.recipient
    whatsapp_config["template_name"] = config.template_name
    whatsapp_config["template_language"] = config.template_language or "en_US"

    whatsapp_config["configured"] = bool(
        whatsapp_config["access_token"]
        and whatsapp_config["phone_number_id"]
        and whatsapp_config["recipient"]
    )
    return {
        "ok": True,
        "token_hint": whatsapp_config["token_hint"],
        "configured": whatsapp_config["configured"]
    }

@app.get("/api/whatsapp/debug")
def debug_whatsapp():
    """Debug endpoint — shows what's saved (token is masked)."""
    token = whatsapp_config["access_token"]
    return {
        "token_length": len(token),
        "token_first_10": token[:10] + "..." if len(token) > 10 else token,
        "token_last_10": "..." + token[-10:] if len(token) > 10 else token,
        "phone_number_id": whatsapp_config["phone_number_id"],
        "recipient": whatsapp_config["recipient"],
        "template_name": whatsapp_config["template_name"],
        "configured": whatsapp_config["configured"],
    }

WA_GRAPH_API_VERSION = "v17.0"

@app.post("/api/whatsapp/test")
async def test_whatsapp():
    if not whatsapp_config["configured"]:
        raise HTTPException(status_code=400, detail="WhatsApp not configured")

    url = f"https://graph.facebook.com/{WA_GRAPH_API_VERSION}/{whatsapp_config['phone_number_id']}/messages"
    headers = {
        "Authorization": f"Bearer {whatsapp_config['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": whatsapp_config["recipient"],
        "type": "text",
        "text": {"body": "Test Message: This is a test notification from the PPE system."}
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return {"ok": True, "details": resp.json()}

@app.post("/api/whatsapp/notify")
async def notify_whatsapp(msg: WhatsAppMessage):
    if not whatsapp_config["configured"]:
        raise HTTPException(status_code=400, detail="WhatsApp not configured")
        
    url = f"https://graph.facebook.com/{WA_GRAPH_API_VERSION}/{whatsapp_config['phone_number_id']}/messages"
    headers = {
        "Authorization": f"Bearer {whatsapp_config['access_token']}",
        "Content-Type": "application/json"
    }
    
    if whatsapp_config["template_name"]:
        parameters = [{"type": "text", "text": str(p)} for p in msg.params]
        payload = {
            "messaging_product": "whatsapp",
            "to": whatsapp_config["recipient"],
            "type": "template",
            "template": {
                "name": whatsapp_config["template_name"],
                "language": {"code": whatsapp_config["template_language"]},
                "components": [
                    {
                        "type": "body",
                        "parameters": parameters
                    }
                ] if parameters else []
            }
        }
    else:
        # Fallback to text message if no template configured (though typically WA business requires templates)
        payload = {
            "messaging_product": "whatsapp",
            "to": whatsapp_config["recipient"],
            "type": "text",
            "text": {"body": msg.message}
        }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return {"ok": True}
