#!/usr/bin/env python3
"""
Wyciąga obrazy twarzy z PDF wraz z tytułami zabiegów.
Użycie: python3 extract_faces.py <plik.pdf>
Wyjście: JSON na stdout [{title, image_base64, page, index}, ...]
"""
import base64
import io
import json
import re
import sys
from collections import defaultdict

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageFilter

MIN_AREA = 20_000
MAX_AREA = 500_000
MIN_RECTANGULARITY = 0.85
MIN_SIDE = 100        # minimalna szerokość i wysokość regionu w px
ASPECT_MIN = 0.6      # min stosunek width/height (nie za wąskie)
ASPECT_MAX = 1.8      # max stosunek width/height (nie za szerokie)
OCR_SCALE = 2
OCR_LANG = "pol"
TITLE_SEARCH_ABOVE = 100
TITLE_SEARCH_INTO = 80
MAX_TITLE_WORDS = 5


def detect_regions(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    # autocontrast żeby podbić jasne/słabe ramki przed detekcją krawędzi
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 10, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = np_img.shape[:2]

    regions, seen = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA < area < MAX_AREA):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x < 5 or y < 5 or (x + bw) > w_img - 5 or (y + bh) > h_img - 5:
            continue
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if not (ASPECT_MIN <= bw / bh <= ASPECT_MAX):
            continue
        if area / (bw * bh) < MIN_RECTANGULARITY:
            continue
        is_dup = any(
            max(0, min(x+bw, sx+sw) - max(x, sx)) * max(0, min(y+bh, sy+sh) - max(y, sy)) > 0.5 * bw * bh
            for sx, sy, sw, sh in seen
        )
        if is_dup:
            continue
        seen.append((x, y, bw, bh))
        regions.append((x, y, bw, bh))

    regions.sort(key=lambda r: r[1])
    return regions


def ocr_full_image(pil_img):
    big = pil_img.resize((pil_img.width * OCR_SCALE, pil_img.height * OCR_SCALE), Image.LANCZOS)
    big = ImageOps.autocontrast(big)
    big = big.filter(ImageFilter.SHARPEN)
    data = pytesseract.image_to_data(big, lang=OCR_LANG, config="--psm 6", output_type=pytesseract.Output.DICT)

    lines = defaultdict(lambda: {"words": [], "tops": [], "bottoms": []})
    for i in range(len(data["text"])):
        if int(data["conf"][i]) < 20 or not data["text"][i].strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines[key]["words"].append(data["text"][i])
        lines[key]["tops"].append(data["top"][i] // OCR_SCALE)
        lines[key]["bottoms"].append((data["top"][i] + data["height"][i]) // OCR_SCALE)

    result = []
    for v in lines.values():
        if not v["words"]:
            continue
        result.append({
            "text": " ".join(v["words"]),
            "y_top": min(v["tops"]),
            "y_bottom": max(v["bottoms"]),
            "word_count": len(v["words"]),
        })
    result.sort(key=lambda l: l["y_top"])
    return result


def find_title(lines, img_y):
    window_top = img_y - TITLE_SEARCH_ABOVE
    window_bottom = img_y + TITLE_SEARCH_INTO

    candidates = [
        l for l in lines
        if window_top <= l["y_top"] <= window_bottom and l["word_count"] <= MAX_TITLE_WORDS
    ]
    if not candidates:
        candidates = [l for l in lines if window_top <= l["y_top"] <= window_bottom]
    if not candidates:
        return "nieznany"

    best = min(candidates, key=lambda l: abs(l["y_top"] - img_y))
    title = best["text"].strip()
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    # usuń artefakty OCR: znaki specjalne, cyfry, wielkie-literowe prefiksy i pojedyncze litery
    title = re.sub(r'^[\W\d_]+', "", title).strip()
    title = re.sub(r'^([A-ZĄĆĘŁŃÓŚŹŻ]+|[a-ząćęłńóśźż])\s+', "", title).strip()
    return title if title else "nieznany"


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    results = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        for img_ref in doc[page_num].get_images():
            xref = img_ref[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            raw = doc.extract_image(xref)
            img_bytes = raw["image"]

            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            regions = detect_regions(np_img)
            if not regions:
                continue

            lines = ocr_full_image(pil_img)

            for idx, (x, y, bw, bh) in enumerate(regions, 1):
                title = find_title(lines, y)
                crop = pil_img.crop((x, y, x + bw, y + bh))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                results.append({
                    "title": title,
                    "page": page_num + 1,
                    "index": idx,
                    "image_base64": base64.b64encode(buf.getvalue()).decode(),
                })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Podaj ścieżkę do pliku PDF"}))
        sys.exit(1)

    results = process_pdf(sys.argv[1])
    print(json.dumps(results, ensure_ascii=False))
