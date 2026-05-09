"""
Wycinanie obrazów twarzy z dokumentu + przypisanie nazw zabiegów przez OCR.
Strategia: OCR całej strony → linie tekstu → dopasowanie tytułu nad każdym obrazem.
Użycie: python3 test_single_mediapipe.py
"""
import io
import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from collections import defaultdict

INPUT_IMAGE = "/home/marcin/Pobrane/pdf_images/page1_img1.png"
OUTPUT_DIR = "/home/marcin/Pobrane/portraits_mediapipe"

MIN_AREA = 20_000
MAX_AREA = 500_000
MIN_RECTANGULARITY = 0.80
OCR_SCALE = 2           # powiększ obraz przed OCR
OCR_LANG = "pol+eng"
TITLE_SEARCH_ABOVE = 100  # szukaj tytułu max tyle px nad górną krawędzią obrazu
TITLE_SEARCH_INTO = 80    # szukaj też wewnątrz obrazu (tytuł może być tuż po krawędzi)
MAX_TITLE_WORDS = 5       # tytuły mają ≤ N słów (dłuższe to opis)


def detect_regions(np_img: np.ndarray) -> list[tuple]:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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


def ocr_full_page(pil_img: Image.Image) -> list[dict]:
    """Zwraca listę linii tekstu: {text, y_top, y_bottom, word_count}"""
    big = pil_img.resize((pil_img.width * OCR_SCALE, pil_img.height * OCR_SCALE), Image.LANCZOS)
    data = pytesseract.image_to_data(big, lang=OCR_LANG, output_type=pytesseract.Output.DICT)

    # grupuj słowa w linie (po page_num+block_num+par_num+line_num)
    lines = defaultdict(lambda: {"words": [], "tops": [], "bottoms": []})
    for i in range(len(data["text"])):
        if int(data["conf"][i]) < 20 or not data["text"][i].strip():
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
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


def find_title(lines: list[dict], img_y: int) -> str:
    """Znajdź tytuł: krótka linia w oknie [img_y - ABOVE, img_y + INTO]."""
    window_top = img_y - TITLE_SEARCH_ABOVE
    window_bottom = img_y + TITLE_SEARCH_INTO

    # najpierw szukaj krótkiej linii w oknie (tytuł)
    candidates = [
        l for l in lines
        if l["y_top"] >= window_top
        and l["y_top"] <= window_bottom
        and l["word_count"] <= MAX_TITLE_WORDS
    ]

    if not candidates:
        # fallback: dowolna linia w oknie
        candidates = [l for l in lines if l["y_top"] >= window_top and l["y_top"] <= window_bottom]

    if not candidates:
        return "nieznany"

    # preferuj linię najbliżej img_y (tuż nad lub tuż po krawędzi obrazu)
    best = min(candidates, key=lambda l: abs(l["y_top"] - img_y))
    return best["text"].strip()


def slugify(name: str, idx: int) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return f"{idx:02d}_{name[:60]}" if name else f"{idx:02d}_nieznany"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pil_img = Image.open(INPUT_IMAGE).convert("RGB")
    np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    print("Wykrywam regiony...")
    regions = detect_regions(np_img)
    print(f"Znaleziono {len(regions)} regionów")

    print("Uruchamiam OCR całej strony...")
    lines = ocr_full_page(pil_img)
    print(f"Wykryto {len(lines)} linii tekstu\n")

    for i, (x, y, bw, bh) in enumerate(regions, 1):
        title = find_title(lines, y)
        print(f"Region {i:02d}: y={y}  tytuł: '{title}'")

        crop = pil_img.crop((x, y, x + bw, y + bh))
        filename = slugify(title, i) + ".png"
        crop.save(os.path.join(OUTPUT_DIR, filename))

    print(f"\nZapisano {len(regions)} obrazów do: {OUTPUT_DIR}")
