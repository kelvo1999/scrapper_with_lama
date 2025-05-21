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
        'coupon': ['img[src*="coupon"]', 'div.entry-content img', 'div.coupon-book img', 'div.post-content img'],
        'hot_buy': ['img[src*="hotbuy"]', 'img[src*="deal"]', 'div.hot-deals img', 'div.post-content img']
    },
    'default_brands': {
        'SunVilla', 'Charmin', 'Yardistry', 'Dyson Cyclone', 'Pistachios', 'Primavera Mistura',
        'Apples', 'Palmiers', 'Waterloo', 'Woozoo', 'Mower', 'Trimmer', 'Jet Blower',
        'Scotts', 'Huggies', 'Powder', 'Cookie', 'Kerrygold', 'Prawn Hacao', 'Kirkland Signature',
        'Samsung', 'Sony', 'Dyson', 'Apple', 'LG', 'Bose', 'Panasonic', 'Starbucks', 'Coca-Cola',
        'Pepsi', 'Tide', 'Bounty', 'Duracell', 'Nestle', 'Kellogg\'s', 'General Mills'
    },
    'exclude_images': [
        'Costco-Insider4.png',
        'logo',
        'banner',
        'header',
        'footer',
        'advertisement',
        'ad',
        'social',
        'date.png',
        'user.png',
        'https://www.facebook.com/tr',
        'folder.png',
        'tag.png'
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
        'url': "https://www.costcoinsider.com/costco-may-and-june-2025-coupon-book/",
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

def split_grid_image_dynamic(img):
    """Dynamically splits a grid image into individual item images."""
    width, height = img.size
    columns = 4 if width > 900 else 3
    rows = 2
    
    cell_width = width // columns
    cell_height = height // rows
    
    sub_images = []
    for row in range(rows):
        for col in range(columns):
            left = col * cell_width
            top = row * cell_height
            right = (col + 1) * cell_width
            bottom = (row + 1) * cell_height
            sub_images.append(img.crop((left, top, right, bottom)))
            
    logging.info(f"Split image into {len(sub_images)} sub-images ({rows} rows, {columns} columns).")
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
You are an expert data extractor for Costco deals. The following text is raw OCR output from a coupon image. It may contain errors (e.g., "Ch4rm1n" instead of "Charmin", "T0il3t P4p3r" instead of "Toilet Paper").

Your task is to clean up the text and extract structured deal data for each item found.

Return a JSON list of dictionaries, where each dictionary represents one item and has the following fields:
- `item_brand`: The brand name of the product. Use the provided list of known brands to correct OCR errors (e.g., "Ch4rm1n" should be "Charmin"). If no brand is explicitly mentioned or clearly inferable from the known brands, leave it empty.
- `item_description`: A clean and concise description of the item, fixing any OCR errors. Max 100 characters.
- `discount`: The exact discount wording as it appears (e.g., "$5 OFF", "20% OFF", "SAVE $10").
- `discount_cleaned`: The numerical value of the discount (e.g., "5", "20", "10"). Extract only the number. If not applicable, leave empty.
- `count_limit`: Any purchase limit (e.g., "Limit 2", "While Supplies Last"). If not specified, leave empty.
- `channel`: Where the deal is available (e.g., "In-Warehouse", "Online", "In-Warehouse + Online", "Warehouse-Only", "Book with Costco"). If not specified, leave empty.
- `discount_period`: The period during which the discount is valid. Extract this from the text if available. If not, infer from the article or leave empty.
- `item_original_price`: The original price of the item, if mentioned (e.g., "$119.99"). Extract only the price including the dollar sign. If not available, leave empty.
- `source_url`: The URL of the image from which this data was extracted. (Set to "{source_url}" for all items extracted from this image).
- `article_name`: The name of the article this coupon belongs to. (Set to "{article_name}" for all items extracted from this image).
- `publish_date`: The publish date of the article. (Set to "{publish_date_str}" for all items extracted from this image).
- `type`: The type of deal. (Set to "{deal_type}" for all items extracted from this image).

**Known Brands for reference:** {known_brands_str}

**Important Rules:**
- Ensure `item_brand` is one of the `known_brands` if a match is found. If the OCR text has a highly confident match (e.g., "Dyson" for "Dyson Cyclone"), use the more specific known brand.
- If a field is missing or cannot be confidently extracted, return an empty string for that field.
- Exclude any non-deal entries, headers, footers, or unrelated text from the output.
- Return ONLY the JSON list, no additional text, explanations, or markdown formatting outside the JSON.

---
OCR Text to Parse:
{raw_text}
---
"""
    response = call_llm(prompt)
    
    try:
        json_start = response.find('[')
        json_end = response.rfind(']')
        
        if json_start == -1 or json_end == -1:
            logging.error("No complete JSON list found in LLM response.")
            logging.debug(f"Raw LLM response: {response}")
            return []
            
        json_string = response[json_start : json_end + 1]
        json_data = json.loads(json_string)
        
        logging.info(f"Extracted {len(json_data)} items from text using LLM.")
        return json_data
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed after LLM call: {e}")
        logging.debug(f"Raw LLM response causing error: {response}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in parse_coupon_data: {e}")
        logging.debug(f"Raw LLM response: {response}")
        return []

def scrape_images_from_page(url, article_name, publish_date_str, is_hot_buy=False):
    """Scrapes images from a given article page, processes them with OCR and LLM."""
    logging.info(f"Scraping images from article: {article_name} ({url})")
    html = get_page(url)
    if not html:
        return []
    
    brand_file = 'hot_buy_brands.txt' if is_hot_buy else 'coupon_book_brands.txt'
    known_brands = load_known_brands(brand_file)
    
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    
    selectors = CONFIG['image_selectors']['hot_buy'] if is_hot_buy else CONFIG['image_selectors']['coupon']
    page_images = []
    for selector in selectors:
        found_images = soup.select(selector)
        if found_images:
            page_images.extend(found_images)
            if not is_hot_buy and any(s in selector for s in ['#coupon-book', 'div.coupon-book']):
                break 
            elif is_hot_buy and any('hotbuy' in s for s in selectors):
                break
    
    if not page_images:
        page_images = soup.select('img')
        logging.warning(f"No specific images found for {url}, falling back to all <img> tags. Found {len(page_images)}.")
    else:
        page_images = list(set(page_images))
        logging.info(f"Final count of images found using selectors: {len(page_images)} for {url}.")

    for img_tag in page_images:
        img_url = img_tag.get('src')
        if not img_url:
            logging.debug(f"Skipping image with no src attribute in {url}")
            continue
            
        if not img_url.startswith('http'):
            img_url = urljoin(url, img_url)
        
        # Log all image URLs for debugging
        logging.debug(f"Found image URL: {img_url}")
        
        if any(exclude.lower() in img_url.lower() for exclude in CONFIG['exclude_images']):
            logging.info(f"Skipping excluded image: {img_url}")
            continue
            
        logging.info(f"Attempting to download and process image: {img_url}")
        img = download_image(img_url, url)
        if not img:
            logging.warning(f"Failed to download or preprocess image: {img_url}")
            continue
        
        if is_hot_buy and img.width > 500 and img.height > 500:
            sub_images = split_grid_image_dynamic(img)
            logging.info(f"Hot buy image split into {len(sub_images)} sub-images.")
        else:
            sub_images = [img]
            logging.info(f"Processing image as a single block (not split for complex layouts).")

        for sub_img in sub_images:
            text = extract_text_from_image(sub_img)
            if not text or len(text.strip()) < 20:
                logging.debug(f"Skipping sub-image with insufficient text (length {len(text.strip())}): {text}")
                continue
            
            parsed_items = parse_coupon_data(text, img_url, article_name, publish_date_str, is_hot_buy, known_brands)
            if parsed_items:
                items.extend(parsed_items)
            else:
                logging.warning(f"LLM returned no parsed items for text block: {text[:100]}...")
    
    if is_hot_buy:
        next_page = soup.find('a', string=re.compile(r'next|›|>', re.IGNORECASE))
        if next_page and next_page.get('href'):
            next_url = urljoin(url, next_page['href'])
            logging.info(f"Following next hot buys page: {next_url}")
            items.extend(scrape_images_from_page(next_url, article_name, publish_date_str, is_hot_buy))
    
    return items

def write_to_csv(items):
    """Writes extracted items to separate CSV files based on their 'type'."""
    fieldnames = [
        "item_brand", "item_description", "discount", "discount_cleaned",
        "count_limit", "channel", "discount_period", "item_original_price",
        "source_url", "article_name", "publish_date", "type"
    ]
    
    coupon_items = [item for item in items if item.get('type') == "Coupon Book"]
    hot_buy_items = [item for item in items if item.get('type') == "Hot Buy"]

    if coupon_items:
        file_exists = os.path.isfile(COUPON_OUTPUT_FILE)
        with open(COUPON_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in coupon_items:
                row_data = {field: item.get(field, "") for field in fieldnames}
                writer.writerow(row_data)
        logging.info(f"Saved {len(coupon_items)} coupon book items to {COUPON_OUTPUT_FILE}")

    if hot_buy_items:
        file_exists = os.path.isfile(HOT_BUYS_OUTPUT_FILE)
        with open(HOT_BUYS_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in hot_buy_items:
                row_data = {field: item.get(field, "") for field in fieldnames}
                writer.writerow(row_data)
        logging.info(f"Saved {len(hot_buy_items)} hot buy items to {HOT_BUYS_OUTPUT_FILE}")

def main():
    """Main function to orchestrate the scraping process."""
    if not initialize():
        logging.error("Scraper initialization failed. Exiting.")
        return

    logging.info("Starting comprehensive Costco coupon scraping.")
    
    for post_info in POSTS_TO_SCRAPE:
        title = post_info['title']
        link = post_info['url']
        is_hot_buy = post_info['is_hot_buy']
        publish_date_str = post_info['publish_date']
        
        logging.info(f"\n--- Processing: {'Hot Buys' if is_hot_buy else 'Coupon Book'} - {title} (Published: {publish_date_str}) ---")
        
        items_from_article = scrape_images_from_page(link, title, publish_date_str, is_hot_buy)
        
        if items_from_article:
            write_to_csv(items_from_article)
        else:
            logging.warning(f"No items extracted from article: {title}")

    logging.info("Done! All specified Costco coupon data scraped and saved to CSV files.")

if __name__ == "__main__":
    main()