# WarehouseSight-AI Backend

GitHub repo name: `StockSight-Backend`

Render service name: `stocksight-backend`

GitHub repo: https://github.com/Lintshiwe/StockSight-Backend

Hugging Face model repo: https://huggingface.co/lintshiwe/Modle_V2.0

Deployed backend URL: https://stocksight-backend-0e6n.onrender.com

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
