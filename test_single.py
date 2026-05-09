"""
Szybki test na jednym obrazie bez uruchamiania serwera.
Użycie: OPENAI_API_KEY=sk-... python3 test_single.py
"""
import sys
sys.path.insert(0, ".")

from main import extract_portraits_from_image
import os

img_path = "/home/marcin/Pobrane/pdf_images/page1_img1.png"

with open(img_path, "rb") as f:
    img_bytes = f.read()

print(f"Wysyłam obraz do GPT-4o ({len(img_bytes)//1024} KB)...")
crops = extract_portraits_from_image(img_bytes)
print(f"Znaleziono {len(crops)} portretów")

os.makedirs("/home/marcin/Pobrane/portraits", exist_ok=True)
for i, crop in enumerate(crops):
    path = f"/home/marcin/Pobrane/portraits/portrait_{i+1}.png"
    with open(path, "wb") as f:
        f.write(crop)
    print(f"  Zapisano: {path}")
