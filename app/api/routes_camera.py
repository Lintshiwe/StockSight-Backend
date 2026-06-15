from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


router = APIRouter()


class CameraSourceRequest(BaseModel):
    source: str


def runtime(request: Request) -> Any:
    return request.app.state.runtime


@router.get("/api/camera/status")
def camera_status(request: Request) -> dict[str, Any]:
    service = runtime(request)
    return {**service.camera.status(), "pipeline_error": service.last_error}


@router.post("/api/camera/start")
def camera_start(request: Request) -> dict[str, Any]:
    return runtime(request).start_camera()


@router.post("/api/camera/stop")
def camera_stop(request: Request) -> dict[str, Any]:
    return runtime(request).stop_camera()


@router.post("/api/camera/source")
def camera_source(payload: CameraSourceRequest, request: Request) -> dict[str, Any]:
    try:
        return runtime(request).set_camera_source(payload.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/stream/frame")
def stream_frame(request: Request) -> StreamingResponse:
    service = runtime(request)

    async def generate():
        while True:
            frame = service.latest_frame_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    service = websocket.app.state.runtime
    try:
        while True:
            await websocket.send_text(json.dumps(service.latest_payload, default=str))
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return

