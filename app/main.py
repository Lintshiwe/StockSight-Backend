from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_analytics, routes_camera, routes_detection, routes_events, routes_model, routes_zones
from app.config import settings
from app.runtime import VisionRuntime
from app.storage.database import Database
from app.storage.repository import Repository


database = Database(settings.database_path)
repository = Repository(database)
runtime = VisionRuntime(repository, settings)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    database.initialize()
    settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=runtime.try_load_configured_model, name="model-autoload", daemon=True).start()
    app.state.database = database
    app.state.repository = repository
    app.state.runtime = runtime


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


app.include_router(routes_camera.router)
app.include_router(routes_detection.router)
app.include_router(routes_events.router)
app.include_router(routes_analytics.router)
app.include_router(routes_model.router)
app.include_router(routes_zones.router)
