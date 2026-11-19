"""FastAPI entrypoint (skeleton only)."""

from fastapi import FastAPI

app = FastAPI(title="aqi-hcho-india API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
