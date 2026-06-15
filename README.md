# WarehouseSight-AI Backend

GitHub repo name: `StockSight-Backend`

Render service name: `stocksight-backend`

Deployed backend URL: pending Render deployment

FastAPI backend for live camera capture, YOLO segmentation inference, object tracking, zone monitoring, event storage, and analytics.

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Place YOLO segmentation weights in `../models/`, then load them with:

```bash
curl -X POST http://localhost:8000/api/model/load \
  -H 'Content-Type: application/json' \
  -d '{"model_path":"yolo11n-seg.pt"}'
```

Start webcam index `0`:

```bash
curl -X POST http://localhost:8000/api/camera/source -H 'Content-Type: application/json' -d '{"source":"0"}'
curl -X POST http://localhost:8000/api/camera/start
```

The MJPEG stream is available at `/api/stream/frame`; JSON detections stream on `/ws/live`.
