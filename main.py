#!/usr/bin/env python3
"""
MiMo Web2API - Biến cookie/session của aistudio.xiaomimimo.com
thành OpenAI-compatible API endpoint.
"""

import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router, setup_static
from app.mimo_client import client_manager

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="MiMo Web2API",
    description="Convert aistudio.xiaomimimo.com web session to OpenAI-compatible API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)
setup_static(app)


@app.on_event("startup")
async def startup():
    logger.info(f"MiMo Web2API starting on {settings.host}:{settings.port}")
    logger.info(f"Loaded {client_manager.count} accounts")
    if client_manager.count == 0:
        logger.warning("No accounts loaded! Add credentials via web UI at /")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )