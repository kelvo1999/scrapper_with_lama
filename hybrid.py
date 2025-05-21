# Hybrid Costco Scraper with LLM & OCR Fallback
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import os
import subprocess
import pytesseract
from PIL import Image, ImageEnhance
from io import BytesIO
import cv2
import numpy as np
import time
import re
import json


# === CONFIG ===
CONFIG = {
    'user_agent': 'Mozilla/5.0',
    'delay': 2,
    'tesseract_path': r'C:\Users\kelvin.shisanya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
    'output_file': 'costco_llm_hybrid_output.csv',
    'llm_model': 'mistral',
}

pytesseract.pytesseract.tesseract_cmd = CONFIG['tesseract_path']


def get_html(url):
    time.sleep(CONFIG['delay'])
    headers = {'User-Agent': CONFIG['user_agent']}
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None


def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.select_one('.entry-content')
    if content:
        text = content.get_text(separator='\n').strip()
        return text if len(text.split()) > 30 else None
    return None


def download_images_from_html(html, referer):
    soup = BeautifulSoup(html, 'html.parser')
    images = soup.select('img')
    valid_imgs = []
    for img_tag in images:
        src = img_tag.get('src')
        # if not src or any(x in src.lower() for x in ['logo', 'header', 'footer']):
        if (
             not src
             or 'facebook.com/tr?' in src
             or any(x in src.lower() for x in ['logo', 'header', 'footer', 'pixel', 'tracker', '.gif'])
            ):

            continue
        if not src.startswith('http'):
            src = requests.compat.urljoin(referer, src)
        valid_imgs.append(src)
    return valid_imgs


def preprocess_image(img_data):
    img = Image.open(BytesIO(img_data))
    img_np = np.array(img)
    if img_np.ndim == 2:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    img = Image.fromarray(denoised)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img


def run_ocr_on_images(image_urls, referer):
    texts = []
    for url in image_urls:
        try:
            res = requests.get(url, headers={'Referer': referer})
            img = preprocess_image(res.content)
            ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6').strip()
            texts.append(ocr_text)
        except Exception as e:
            print(f"⚠️ OCR failed for {url}: {e}")
    return '\n'.join(texts)


def call_llm(prompt):
    try:
        result = subprocess.run([
            "ollama", "run", CONFIG['llm_model']
        ], input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode()
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return ""


def extract_items_with_llm(raw_text):
    prompt = f"""
Extract all Costco coupon or hot buy deals from the text below. Return a JSON list with fields: 
item_name, description, discount_value, channel, validity_period, item_limit, type.

TEXT:
---
{raw_text}
---
"""
    response = call_llm(prompt)
    try:
        json_start = response.find('[')
        items = json.loads(response[json_start:])
        return items
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        return []


def save_items_to_csv(items):
    fields = ["item_name", "description", "discount_value", "channel", "validity_period", "item_limit", "type"]
    file_exists = os.path.exists(CONFIG['output_file'])
    with open(CONFIG['output_file'], 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for item in items:
            writer.writerow(item)


def process_costco_page(url):
    print(f"🌐 Processing: {url}")
    html = get_html(url)
    if not html:
        return
    raw_text = extract_text_from_html(html)
    if not raw_text:
        print("🔍 No HTML text found — falling back to OCR.")
        image_urls = download_images_from_html(html, url)
        raw_text = run_ocr_on_images(image_urls, url)
    if not raw_text:
        print("⚠️ No data to process.")
        return
    items = extract_items_with_llm(raw_text)
    if items:
        save_items_to_csv(items)
        print(f"✅ Saved {len(items)} items from {url}")
    else:
        print("⚠️ LLM returned no valid items.")


def main():
    urls = [
        "https://www.costcoinsider.com/costco-april-2025-coupon-book/",
        "https://www.costcoinsider.com/costco-april-2025-hot-buys-coupons/"
    ]
    for url in urls:
        process_costco_page(url)


if __name__ == "__main__":
    main()








