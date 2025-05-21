import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
from urllib.parse import urljoin
import time
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
from io import BytesIO
from rapidfuzz import process
import logging
import os
import subprocess
import json

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper_log.txt"),
        logging.StreamHandler()
    ]
)

# === CONFIGURATION ===
CONFIG = {
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'delay': 2,
    'tesseract_path': r'C:\Users\kelvin.shisanya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
    'image_selectors': {
        'coupon': ['img[src*="coupon"]', 'div.entry-content img'],
        'hot_buy': ['img[src*="hotbuy"]', 'img[src*="deal"]', 'div.hot-deals img']
    },
    'default_brands': {
        'SunVilla', 'Charmin', 'Yardistry', 'Dyson Cyclone', 'Pistachios', 'Primavera Mistura',
        'Apples', 'Palmiers', 'Waterloo', 'Woozoo', 'Mower', 'Trimmer', 'Jet Blower',
        'Scotts', 'Huggies', 'Powder', 'Cookie', 'Kerrygold', 'Prawn Hacao'
    },
    'exclude_images': [
        'Costco-Insider4.png',
        'Costco-May-2025-Hot-Buys-Coupons-Cover.jpg',
        'logo',
        'banner',
        'header',
        'footer'
    ]
}

COUPON_OUTPUT_FILE = "coupon_books.csv"
HOT_BUYS_OUTPUT_FILE = "hot_buys.csv"

def load_known_brands(filepath):
    """Load known brands from a file into a set, fallback to default brands if file missing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            brands = set(line.strip() for line in f if line.strip())
            if brands:
                return brands
            else:
                logging.warning(f"Brand file {filepath} is empty. Using default brands.")
                return CONFIG['default_brands']
    except FileNotFoundError:
        logging.warning(f"Brand file {filepath} not found. Using default brands.")
        return CONFIG['default_brands']

def initialize():
    try:
        pytesseract.pytesseract.tesseract_cmd = CONFIG['tesseract_path']
        pytesseract.get_tesseract_version()
        logging.info("Tesseract initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Tesseract init error: {e}")
        return False

def get_page(url):
    try:
        time.sleep(CONFIG['delay'])
        headers = {'User-Agent': CONFIG['user_agent']}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        logging.info(f"Successfully fetched {url}")
        return res.text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def download_image(img_url, referer):
    try:
        headers = {'User-Agent': CONFIG['user_agent'], 'Referer': referer}
        res = requests.get(img_url, headers=headers)
        res.raise_for_status()
        if not res.headers.get("Content-Type", "").startswith("image"):
            logging.warning(f"Skipping non-image: {img_url}")
            return None
        img = Image.open(BytesIO(res.content))
        img_np = np.array(img)

        scale_factor = 2
        img_np = cv2.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        if img_np.ndim == 2:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        img = Image.fromarray(sharpened)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        logging.info(f"Successfully downloaded and processed image: {img_url}")
        return img
    except Exception as e:
        logging.error(f"Error processing image {img_url}: {e}")
        return None

def extract_text_from_image(img):
    config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
    try:
        text = pytesseract.image_to_string(img, config=config).strip()
        ocr_corrections = {
            '|': 'I',
            '0': 'O',
            'vv': 'W',
            '1': 'I',
            '5': 'S',
            '$': '$',
            '\n': ' ',
            '\t': ' ',
            '  ': ' '
        }
        for wrong, right in ocr_corrections.items():
            text = text.replace(wrong, right)
        text = re.sub(r'[^a-zA-Z0-9\s$/.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Extracted text from image: {text[:100]}...")
        return text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def call_llm(prompt, model="mistral"):
    """Call local LLM using Ollama CLI (ensure model is pulled)."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        output = result.stdout.decode().strip()
        if result.stderr:
            logging.warning(f"LLM stderr: {result.stderr.decode()}")
        logging.info(f"LLM response length: {len(output)} characters")
        return output
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return ""

def parse_coupon_data(raw_text, is_hot_buy, known_brands):
    """Use the LLM to extract structured item info from OCR text."""
    deal_type = "Hot Buy" if is_hot_buy else "Coupon Book"
    validity_period = "4/28/25 - 5/11/25" if is_hot_buy else "5/14/25 - 6/8/25"
    known_brands_str = ", ".join(known_brands)

    prompt = f"""
You are an expert data extractor for Costco deals. The following text is raw OCR output from a coupon image, which may contain errors (e.g., "Ch4rm1n" instead of "Charmin"). Your task is to clean up the text and extract structured deal data. Use the list of known brands to help identify the correct brand for the item_name. Return a JSON list with these fields for each item:
- item_name (product name, clean and concise, max 50 characters, should match a known brand if possible)
- description (brief item description, max 100 characters, clean and free of OCR errors)
- discount_value (discount amount, e.g., "$5 OFF" or "$10", clean format)
- channel (e.g., "In-Warehouse", "Online", "In-Warehouse + Online", "Warehouse-Only", or "Book with Costco")
- validity_period (set to "{validity_period}" for all items)
- item_limit (e.g., "Limit 2" or "While Supplies Last", or empty if not specified)
- type (set to "{deal_type}" for all items)

Rules:
- Use the list of known brands: {known_brands_str}
- If the OCR text contains a brand that closely matches a known brand (e.g., "Ch4rm1n" for "Charmin"), set item_name to the correct brand ("Charmin").
- Clean up the description by fixing OCR errors (e.g., "T0il3t P4p3r" should be "Toilet Paper").
- For discount_value, extract only the discount (e.g., "$5 OFF", "$10"), removing extra text like "(While Supplies Last)".
- If a field is missing, return an empty string for that field.
- Exclude non-deal entries like headers, footers, or unrelated text.
- Return ONLY the JSON list, no additional text or explanations.

---
{raw_text}
---
"""
    response = call_llm(prompt)
    try:
        json_start = response.find('[')
        if json_start == -1:
            logging.error("No JSON list found in LLM response")
            return []
        json_data = json.loads(response[json_start:])
        logging.info(f"Extracted {len(json_data)} items from text")
        return json_data
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
        logging.debug(f"Raw LLM response: {response}")
        return []

def scrape_images_from_page(url, is_hot_buy=False):
    html = get_page(url)
    if not html:
        return []
    brand_file = 'hot_buy_brands.txt' if is_hot_buy else 'coupon_book_brands.txt'
    known_brands = load_known_brands(brand_file)
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    if not is_hot_buy:
        coupon_container = soup.select_one('#coupon-book')
        images = coupon_container.select('img') if coupon_container else []
        logging.info(f"Found {len(images)} coupon book images in #coupon-book")
    else:
        selectors = CONFIG['image_selectors']['hot_buy']
        images = []
        for selector in selectors:
            images = soup.select(selector)
            if images:
                break
        else:
            images = soup.select('img')
        logging.info(f"Found {len(images)} hot buy images")

    for img_tag in images:
        img_url = img_tag.get('src')
        if not img_url:
            continue
            
        if not img_url.startswith('http'):
            img_url = urljoin(url, img_url)
        
        if any(exclude.lower() in img_url.lower() for exclude in CONFIG['exclude_images']):
            logging.info(f"Skipping non-product image: {img_url}")
            continue
            
        logging.info(f"Downloading image: {img_url}")
        img = download_image(img_url, url)
        if not img:
            continue
        text = extract_text_from_image(img)
        if not text:
            continue
        parsed = parse_coupon_data(text, is_hot_buy, known_brands)
        if isinstance(parsed, list):
            items.extend(parsed)
    
    if is_hot_buy:
        next_page = soup.find('a', string=re.compile(r'next|›|>', re.IGNORECASE))
        if next_page and next_page.get('href'):
            next_url = urljoin(url, next_page['href'])
            logging.info(f"Following next page: {next_url}")
            items.extend(scrape_images_from_page(next_url, is_hot_buy))
    return items

def write_to_csv(items):
    """Write items to separate CSV files based on type."""
    fieldnames = ["item_name", "description", "discount_value", "channel", "validity_period", "item_limit", "type"]
    
    coupon_items = [item for item in items if item.get('type') == "Coupon Book"]
    hot_buy_items = [item for item in items if item.get('type') in ["Hot Buy", "Travel Hot Buy"]]

    if coupon_items:
        file_exists = os.path.isfile(COUPON_OUTPUT_FILE)
        with open(COUPON_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in coupon_items:
                writer.writerow(item)
        logging.info(f"Saved {len(coupon_items)} coupon book items to {COUPON_OUTPUT_FILE}")

    if hot_buy_items:
        file_exists = os.path.isfile(HOT_BUYS_OUTPUT_FILE)
        with open(HOT_BUYS_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in hot_buy_items:
                writer.writerow(item)
        logging.info(f"Saved {len(hot_buy_items)} hot buy items to {HOT_BUYS_OUTPUT_FILE}")
# one night in the smoky hut of a
def main():
    if not initialize(): 
        return
    logging.info("Scraping Coupon Book...")
    coupon_url = "https://www.costcoinsider.com/costco-may-and-june-2025-coupon-book/"
    coupons = scrape_images_from_page(coupon_url, is_hot_buy=False)
    write_to_csv(coupons)
    logging.info("Scraping Hot Buys...")
    hot_buys_url = "https://www.costcoinsider.com/costco-may-2025-hot-buys-coupons/"
    hotbuys = scrape_images_from_page(hot_buys_url, is_hot_buy=True)
    write_to_csv(hotbuys)
    logging.info("Done! Check the CSV files.")

if __name__ == "__main__":
    main()