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
    # 'image_selectors': {
    #     'coupon': ['div.entry-content img.size-full', 'div.coupon-book img.size-full'],
    #     'hot_buy': ['div.hot-deals img.size-full', 'div.post-content img.size-full']
    },
    'pagination_selectors': {
        'coupon': 'div.pagination a[href*="page-"]',
        'hot_buy': 'div.pagination a[href*="page-"]'
    },
    'default_brands': {
        'SunVilla', 'Charmin', 'Yardistry', 'Dyson Cyclone', 'Pistachios', 'Primavera Mistura',
        'Apples', 'Palmiers', 'Waterloo', 'Woozoo', 'Mower', 'Trimmer', 'Jet Blower',
        'Scotts', 'Huggies', 'Powder', 'Cookie', 'Kerrygold', 'Prawn Hacao', 'Kirkland Signature',
        'Samsung', 'Sony', 'Dyson', 'Apple', 'LG', 'Bose', 'Panasonic', 'Starbucks', 'Coca-Cola',
        'Pepsi', 'Tide', 'Bounty', 'Duracell', 'Nestle', 'Kellogg\'s', 'General Mills'
    },
    # 'exclude_images': [
    #     'Costco-Insider4.png',
    #     'logo',
    #     'banner',
    #     'header',
    #     'footer',
    #     'advertisement',
    #     'ad',
    #     'social',
    #     'date.png',
    #     'user.png',
    #     'https://www.facebook.com/tr',
    #     'folder.png',
    #     'tag.png'
    # ],
     'exclude_images': [
        'Costco-Insider4.png',
        'Costco-April-2025-Hot-Buys-Coupons-Cover.jpg',
        'logo',
        'banner',
        'header',
        'footer'
    ],
    'history_years': 2
}

# Output CSV file names
COUPON_OUTPUT_FILE = "coupon_books.csv"
HOT_BUYS_OUTPUT_FILE = "hot_buys.csv"

# Predefined list of articles to scrape
POSTS_TO_SCRAPE = [
    {
        'title': "Costco May and June 2025 Coupon Book",
        'url': "https://www.costcoinsider.com/costco-may-and-june-2025-coupon-book/#coupon-book",
        'is_hot_buy': False,
        'publish_date': "2025-05-13 00:00:00"
    },
    {
        'title': "Costco May 2025 Hot Buys Coupons",
        'url': "https://www.costcoinsider.com/costco-may-2025-hot-buys-coupons/",
        'is_hot_buy': True,
        'publish_date': "2025-04-27 00:00:00"
    }
]

def load_known_brands(filepath):
    """Load known brands from a file into a set, fallback to default brands if file missing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            brands = set(line.strip() for line in f if line.strip())
            if brands:
                logging.info(f"Loaded {len(brands)} brands from {filepath}")
                return brands
            else:
                logging.warning(f"Brand file {filepath} is empty. Using default brands.")
                return CONFIG['default_brands']
    except FileNotFoundError:
        logging.warning(f"Brand file {filepath} not found. Using default brands.")
        return CONFIG['default_brands']

def initialize():
    """Initializes Tesseract OCR engine."""
    try:
        pytesseract.pytesseract.tesseract_cmd = CONFIG['tesseract_path']
        pytesseract.get_tesseract_version()
        logging.info("Tesseract initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Tesseract init error: {e}. Please ensure Tesseract is installed and the path is correct.")
        return False

def get_page(url):
    """Fetches the HTML content of a given URL."""
    try:
        time.sleep(CONFIG['delay'])
        headers = {'User-Agent': CONFIG['user_agent']}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        logging.info(f"Successfully fetched {url}")
        return res.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def download_image(img_url, referer):
    """Downloads an image, preprocesses it for OCR, and returns a PIL Image object."""
    try:
        headers = {'User-Agent': CONFIG['user_agent'], 'Referer': referer}
        res = requests.get(img_url, headers=headers)
        res.raise_for_status()
        if not res.headers.get("Content-Type", "").startswith("image"):
            logging.warning(f"Skipping non-image content: {img_url}")
            return None
        
        img = Image.open(BytesIO(res.content))
        img_np = np.array(img)

        # Enhanced preprocessing
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
        
        logging.info(f"Successfully downloaded and preprocessed image: {img_url}")
        return img
    except Exception as e:
        logging.error(f"Error processing image {img_url}: {e}")
        return None

def split_grid_image(img, rows=2, cols=2):
    """Splits image into grid with specified rows/columns"""
    width, height = img.size
    cell_width = width // cols
    cell_height = height // rows
    
    sub_images = []
    for row in range(rows):
        for col in range(cols):
            left = col * cell_width
            top = row * cell_height
            right = (col + 1) * cell_width
            bottom = (row + 1) * cell_height
            sub_images.append(img.crop((left, top, right, bottom)))
    
    logging.info(f"Split image into {rows}x{cols} grid ({len(sub_images)} sub-images)")
    return sub_images

def extract_text_from_image(img):
    """Extracts text from an image using Tesseract OCR and applies corrections."""
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    try:
        text = pytesseract.image_to_string(img, config=config).strip()
        ocr_corrections = {
            '|': 'I',
            '0': 'O',
            'vv': 'W',
            '1': 'I',
            '5': 'S',
        }
        for wrong, right in ocr_corrections.items():
            text = text.replace(wrong, right)
        
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s$/.,-]', '', text)
        
        logging.info(f"Extracted text from image (first 100 chars): {text[:100]}...")
        return text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def call_llm(prompt, model="mistral"):
    """Calls a local LLM using Ollama CLI to get a structured response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        output = result.stdout.decode('utf-8').strip()
        if result.stderr:
            logging.warning(f"LLM stderr: {result.stderr.decode('utf-8')}")
        logging.info(f"LLM response length: {len(output)} characters")
        return output
    except FileNotFoundError:
        logging.error("Ollama command not found. Please ensure Ollama is installed and in your PATH.")
        return ""
    except subprocess.TimeoutExpired:
        logging.error("LLM call timed out after 120 seconds.")
        return ""
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return ""

def parse_coupon_data(raw_text, source_url, article_name, publish_date_str, is_hot_buy, known_brands):
    """Uses the LLM to extract structured item information from OCR text."""
    deal_type = "Hot Buy" if is_hot_buy else "Coupon Book"
    known_brands_str = ", ".join(known_brands)

    prompt = f"""
Extract Costco deal data from this OCR text and return as JSON:
{raw_text}

Required fields per item:
- item_brand (from known brands: {known_brands_str})
- item_description (max 100 chars)
- discount (exact text like "$5 OFF")
- discount_cleaned (just the number)
- count_limit (if mentioned)
- channel (where valid)
- discount_period
- item_original_price
- source_url: "{source_url}"
- article_name: "{article_name}"
- publish_date: "{publish_date_str}"
- type: "{deal_type}"

Return ONLY valid JSON array of items, no other text.
"""
    response = call_llm(prompt)
    
    try:
        # Extract JSON from response
        json_start = response.find('[')
        json_end = response.rfind(']')
        if json_start == -1 or json_end == -1:
            return []
            
        json_string = response[json_start:json_end+1]
        return json.loads(json_string)
    except json.JSONDecodeError:
        logging.error("Failed to parse LLM response as JSON")
        return []

def scrape_images_from_page(url, article_name, publish_date_str, is_hot_buy=False):
    """Scrapes images from a given article page, including pagination."""
    logging.info(f"Scraping {'hot buys' if is_hot_buy else 'coupon book'} from: {url}")
    html = get_page(url)
    if not html:
        return []
    
    brand_file = 'hot_buy_brands.txt' if is_hot_buy else 'coupon_book_brands.txt'
    known_brands = load_known_brands(brand_file)
    
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    
    # Find all relevant images
    selectors = CONFIG['image_selectors']['hot_buy'] if is_hot_buy else CONFIG['image_selectors']['coupon']
    images = []
    for selector in selectors:
        images.extend(soup.select(selector))
    
    logging.info(f"Found {len(images)} images on page")
    
    # Process each image
    for img_tag in images:
        img_url = img_tag.get('src') or img_tag.get('data-src')
        if not img_url:
            continue
            
        img_url = urljoin(url, img_url)
        
        # Skip excluded images
        if any(exclude.lower() in img_url.lower() for exclude in CONFIG['exclude_images']):
            continue
            
        # Download and process image
        img = download_image(img_url, url)
        if not img:
            continue
            
        # Split image based on deal type
        rows = 2
        cols = 4 if is_hot_buy else 2  # Hot buys typically have 4 columns
        
        try:
            sub_images = split_grid_image(img, rows, cols)
        except Exception as e:
            logging.error(f"Failed to split image: {e}")
            sub_images = [img]  # Fallback to whole image
            
        # Process each sub-image
        for sub_img in sub_images:
            text = extract_text_from_image(sub_img)
            if len(text.strip()) < 20:
                continue
                
            parsed_items = parse_coupon_data(text, img_url, article_name, publish_date_str, is_hot_buy, known_brands)
            if parsed_items:
                items.extend(parsed_items)
    
    # Handle pagination
    pagination_selector = CONFIG['pagination_selectors']['hot_buy'] if is_hot_buy else CONFIG['pagination_selectors']['coupon']
    page_links = soup.select(pagination_selector)
    
    if page_links:
        logging.info(f"Found {len(page_links)} pagination links")
        for page_link in page_links:
            page_url = urljoin(url, page_link['href'])
            if page_url != url:  # Avoid infinite loops
                items.extend(scrape_images_from_page(page_url, article_name, publish_date_str, is_hot_buy))
    
    return items

def write_to_csv(items):
    """Writes extracted items to appropriate CSV files."""
    fieldnames = [
        "item_brand", "item_description", "discount", "discount_cleaned",
        "count_limit", "channel", "discount_period", "item_original_price",
        "source_url", "article_name", "publish_date", "type"
    ]
    
    # Separate items by type
    coupon_items = [item for item in items if item.get('type') == "Coupon Book"]
    hot_buy_items = [item for item in items if item.get('type') == "Hot Buy"]

    # Write coupon book items
    if coupon_items:
        file_exists = os.path.isfile(COUPON_OUTPUT_FILE)
        with open(COUPON_OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(coupon_items)
        logging.info(f"Saved {len(coupon_items)} coupon items to {COUPON_OUTPUT_FILE}")

    # Write hot buy items
    if hot_buy_items:
        file_exists = os.path.isfile(HOT_BUYS_OUTPUT_FILE)
        with open(HOT_BUYS_OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(hot_buy_items)
        logging.info(f"Saved {len(hot_buy_items)} hot buy items to {HOT_BUYS_OUTPUT_FILE}")

def main():
    """Main execution function."""
    if not initialize():
        return

    logging.info("Starting Costco coupon scraper")
    
    all_items = []
    for post in POSTS_TO_SCRAPE:
        logging.info(f"\nProcessing: {post['title']}")
        items = scrape_images_from_page(
            post['url'],
            post['title'],
            post['publish_date'],
            post['is_hot_buy']
        )
        if items:
            all_items.extend(items)
    
    if all_items:
        write_to_csv(all_items)
        logging.info(f"Total items saved: {len(all_items)}")
    else:
        logging.warning("No items were extracted from any pages")

if __name__ == "__main__":
    main()