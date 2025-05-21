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

        # Resize image to improve OCR
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

def fuzzy_find_brand(ocr_text, known_brands):
    """Fuzzy match OCR output to known brands with a lower threshold."""
    match = None
    score = 0
    if known_brands:
        # Clean the OCR text for better matching
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', ocr_text).strip().lower()
        match_tuple = process.extractOne(cleaned_text, known_brands, score_cutoff=70)  # Lowered threshold
        if match_tuple:
            match, score, _ = match_tuple
            logging.info(f"Fuzzy matched '{cleaned_text}' to '{match}' with score {score}")
    return match

def parse_coupon_data(text, source_url, is_hot_buy, known_brands):
    blocks = re.split(r'(?=\$\d+(?:\.\d{2})?\s*OFF)', text)
    data = []
    deal_type = "Hot Buy" if is_hot_buy else "Coupon Book"
    validity_period = "4/28/25 - 5/11/25" if is_hot_buy else "5/14/25 - 6/8/25"

    for block in blocks:
        lines = [line.strip() for line in block.split(' ') if line.strip()]
        if not lines:
            continue

        full_text = ' '.join(lines)

        if len(full_text) < 10 or re.search(r'BOOK WITH|TRAVEL|PACKAGE|^\W+$', full_text, re.IGNORECASE):
            continue

        # --- Brand Extraction for item_name ---
        item_name = ""
        # First try exact match
        for brand in known_brands:
            if re.search(rf'\b{re.escape(brand.lower())}\b', full_text.lower()):
                item_name = brand
                logging.info(f"Exact brand match: {brand}")
                break

        # If no exact match, try fuzzy matching
        if not item_name:
            possible_brand = ' '.join(full_text.split()[:3])
            fuzzy_brand = fuzzy_find_brand(possible_brand, known_brands)
            if fuzzy_brand:
                item_name = fuzzy_brand
            else:
                # Fallback: try matching the first word
                first_word = full_text.split()[0] if full_text.split() else ""
                fuzzy_brand = fuzzy_find_brand(first_word, known_brands)
                if fuzzy_brand:
                    item_name = fuzzy_brand
                else:
                    brand_match = re.match(r'^([A-Z][a-zA-Z0-9&\-\']+)', full_text)
                    if brand_match:
                        item_name = brand_match.group(1)

        if any(char.isdigit() for char in item_name):
            item_name = ""

        if not item_name:
            item_name = ' '.join(full_text.split()[:3])[:50]

        # --- Description Extraction ---
        description = full_text.strip()
        if item_name:
            pattern = re.compile(rf'^{re.escape(item_name)}[\s:,-]*', re.IGNORECASE)
            description = pattern.sub('', description).strip()

        remove_patterns = [
            r'\$[0-9]+(?:\.\d{2})?\s*(?:OFF|off|Save|SAVE)?',
            r'Limit\s+\d+|While\s+supplies\s+last|Qty\s+Limit\s+\d+|Limit:\s+\d+',
            r'warehouse|online|only|in-warehouse',
            r'[^\w\s]'
        ]
        for pat in remove_patterns:
            description = re.sub(pat, '', description, flags=re.IGNORECASE)

        description = re.sub(r'\s+', ' ', description).strip()
        description = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', description)
        description = description[:100]

        # --- Discount and Limit Extraction ---
        discount_match = re.search(r'\$[0-9]+(?:\.\d{2})?\s*OFF', text, re.IGNORECASE)
        discount_value = discount_match.group(0) if discount_match else ""

        # Check if limit is in discount_value and extract it
        limit_in_discount = re.search(r'(Limit\s+\d+|While\s+supplies\s+last|Qty\s+Limit\s+\d+|Limit:\s+\d+)', discount_value, re.IGNORECASE)
        if limit_in_discount:
            item_limit = limit_in_discount.group(0)
            discount_value = re.sub(item_limit, '', discount_value).strip()
        else:
            # Try finding limit in the full text
            limit = re.search(r'(Limit\s+\d+|While\s+supplies\s+last|Qty\s+Limit\s+\d+|Limit:\s+\d+)', text, re.IGNORECASE)
            item_limit = limit.group(0) if limit else ""

        # --- Channel Extraction ---
        channel = ""
        if is_hot_buy:
            warehouse = 'warehouse' in text.lower()
            online = 'online' in text.lower()
            if warehouse and online:
                channel = "In-Warehouse + Online"
            elif warehouse:
                channel = "In-Warehouse"
            elif online:
                channel = "Online"

        row = {
            'item_name': item_name,
            'description': description,
            'discount_value': discount_value,
            'channel': channel,
            'validity_period': validity_period,
            'item_limit': item_limit,
            'type': deal_type
        }
        data.append(row)
    return data

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
        parsed = parse_coupon_data(text, img_url, is_hot_buy, known_brands)
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