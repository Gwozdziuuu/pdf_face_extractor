import json
import tempfile
import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from extract_faces import process_pdf

app = FastAPI()


@app.post("/extract")
async def extract(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Wymagany plik PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = process_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return JSONResponse(results)


@app.get("/health")
def health():
    return {"status": "ok"}
