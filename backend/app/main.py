"""FastAPI entrypoint.

Created by the tutor branch because something had to mount a router first.
Other branches: add one `include_router` line below — please don't restructure
this file, it is the most conflict-prone file in the repo.

    uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

# Must run BEFORE the router import: importing it constructs the LLM client, which
# reads the API key and model names from the environment.
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.tutor.router import router as tutor_router  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Bodhi", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon demo; tighten before anything real
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tutor_router)
# app.include_router(ocr_router)
# app.include_router(rag_router)
# app.include_router(teachback_router)
# app.include_router(practice_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
