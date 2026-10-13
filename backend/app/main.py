from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import chat, collections, documents, settings, system
from .config import get_settings
from .providers import LLMError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="PDF Analyst API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


for r in (system.router, documents.router, collections.router, chat.router, settings.router):
    app.include_router(r)
