import base64
import io
import json
import re
import zipfile

import fitz
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from PIL import Image

app = FastAPI()
client = OpenAI()  # wymaga OPENAI_API_KEY w env

PROMPT = (
    "This image contains a grid of individual portrait photos of people. "
    "Find every distinct portrait photo and return ONLY a JSON array with pixel bounding boxes. "
    "Each box should tightly crop one person. "
    "Format: [{\"x1\": int, \"y1\": int, \"x2\": int, \"y2\": int}]. "
    "No explanation, no markdown, just raw JSON array."
)


def parse_boxes(raw: str) -> list[dict]:
    raw = raw.strip()
    # usuń ewentualne ```json ... ``` opakowanie
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def extract_portraits_from_image(img_bytes: bytes) -> list[bytes]:
    b64 = base64.b64encode(img_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high"
                }}
            ]
        }],
        max_tokens=2000
    )

    raw = response.choices[0].message.content
    boxes = parse_boxes(raw)

    pil_img = Image.open(io.BytesIO(img_bytes))
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        # upewnij się że koordynaty są w granicach obrazu
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(pil_img.width, x2)
        y2 = min(pil_img.height, y2)
        if x2 > x1 and y2 > y1:
            crop = pil_img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crops.append(buf.getvalue())
    return crops


@app.post("/extract-portraits")
async def extract_portraits(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Prześlij plik PDF")

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    zip_buffer = io.BytesIO()
    total = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for page_num in range(len(doc)):
            page_imgs = doc[page_num].get_images()
            for img_idx, img_ref in enumerate(page_imgs):
                xref = img_ref[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                try:
                    crops = extract_portraits_from_image(img_bytes)
                except Exception as e:
                    # jeśli AI nie zwróciło poprawnego JSON, zapisz surowy obraz
                    zf.writestr(
                        f"page{page_num+1}_raw{img_idx+1}.png",
                        img_bytes
                    )
                    continue

                for crop_idx, crop_bytes in enumerate(crops):
                    filename = f"page{page_num+1}_portrait{total + crop_idx + 1}.png"
                    zf.writestr(filename, crop_bytes)

                total += len(crops)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=portraits.zip"}
    )


@app.get("/health")
def health():
    return {"status": "ok"}
