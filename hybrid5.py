import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
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
from rapidfuzz import process # Ensure rapidfuzz is imported

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG for more detailed output
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
        'coupon': ['img[src*="coupon"]', 'div.entry-content img', 'div.coupon-book img', 'div.post-content img', 'article img', 'div img'],
        'hot_buy': ['img[src*="hotbuy"]', 'img[src*="deal"]', 'div.hot-deals img', 'div.post-content img', 'article img', 'div img']
    },
    'pagination_selectors': {
        'category_older': 'div.alignleft a', # Selector for "Older Entries" on category pages
        'category_newer': 'div.alignright a', # Selector for "Newer Entries" on category pages (not used for historical scrape)
        'article_next_page': 'a[href*="page-"], a[href*="next"], a.next' # For multi-page articles (e.g., hot buys)
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
        'logo.', # Added dot to ensure it matches "logo.png", "logo.jpg" etc.
        'banner.',
        'header.',
        'footer.',
        'advertisement.',
        'ad.',
        'social.',
        'date.png',
        'user.png',
        'folder.png',
        'tag.png',
        'https://www.facebook.com/tr', # Specific Facebook tracking pixel URL
        '-Cover.jpg' # Catches cover images like "Costco-April-2025-Hot-Buys-Coupons-Cover.jpg"
    ],
    'history_years': 2,
    'max_article_pages': 10 # Max pages to scrape within a single article (e.g., for hot buys pagination)
}

# Output CSV file names
COUPON_OUTPUT_FILE = "coupon_books.csv"
HOT_BUYS_OUTPUT_FILE = "hot_buys.csv"

def load_known_brands(filepath):
    """Load known brands from a file into a set, fallback to default brands if file missing."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                brands = set(line.strip() for line in f if line.strip())
                if brands:
                    logging.info(f"Loaded {len(brands)} brands from {filepath}")
                    return brands
                else:
                    logging.warning(f"Brand file {filepath} is empty. Using default brands.")
                    return CONFIG['default_brands']
        else:
            logging.warning(f"Brand file {filepath} not found. Using default brands.")
            return CONFIG['default_brands']
    except Exception as e:
        logging.error(f"Error loading known brands from {filepath}: {e}. Using default brands.")
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
        logging.debug(f"Fetching URL: {url}")
        time.sleep(CONFIG['delay'])
        headers = {'User-Agent': CONFIG['user_agent']}
        res = requests.get(url, headers=headers, timeout=15) # Added timeout
        res.raise_for_status()
        logging.info(f"Successfully fetched {url}")
        return res.text
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def download_image(img_url, referer):
    """Downloads an image, preprocesses it for OCR, and returns a PIL Image object."""
    try:
        logging.debug(f"Downloading image: {img_url} (Referer: {referer})")
        headers = {'User-Agent': CONFIG['user_agent'], 'Referer': referer}
        res = requests.get(img_url, headers=headers, stream=True, timeout=15) # Added timeout
        res.raise_for_status()

        content_type = res.headers.get('Content-Type', '').lower()
        
        # Fallback: infer content type from file extension if header is missing or generic
        if not content_type or not content_type.startswith('image'):
            parsed_url = urlparse(img_url)
            filename = os.path.basename(parsed_url.path).lower()
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')): # Added .webp
                content_type = 'image/' + filename.split('.')[-1]
            else:
                logging.warning(f"No valid Content-Type header and unknown file extension for {img_url}. Skipping.")
                return None

        if not content_type.startswith('image'):
            logging.warning(f"Skipping non-image content: {img_url} (Content-Type: {content_type})")
            return None
        
        # Read content into BytesIO for PIL
        img_bytes = BytesIO(res.content)
        img = Image.open(img_bytes)
        
        # Convert to numpy array for OpenCV processing
        img_np = np.array(img)

        # Upscale image for better OCR accuracy
        scale_factor = 2
        # Ensure image is not empty before resizing
        if img_np.size == 0:
            logging.warning(f"Downloaded image {img_url} is empty. Skipping preprocessing.")
            return None
        img_np = cv2.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Convert to BGR if not already (for OpenCV processing)
        if img_np.ndim == 2: # Grayscale
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4: # RGBA
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else: # RGB
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Convert back to PIL Image for Tesseract
        img = Image.fromarray(sharpened)
        
        # Enhance contrast and sharpness (PIL)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        
        logging.info(f"Successfully downloaded and preprocessed image: {img_url}")
        return img
    except RecursionError as e:
        logging.error(f"RecursionError while downloading {img_url}: {e}. This might indicate a problem with the image file or server.")
        return None
    except Exception as e:
        logging.error(f"Error processing image {img_url}: {e}")
        return None

def split_grid_image(img, rows, cols):
    """Splits image into grid with specified rows/columns."""
    width, height = img.size
    
    if width == 0 or height == 0:
        logging.warning("Attempted to split an image with zero width or height. Returning original image.")
        return [img]

    cell_width = width // cols
    cell_height = height // rows
    
    if cell_width == 0 or cell_height == 0:
        logging.warning(f"Calculated cell dimensions are zero (width={cell_width}, height={cell_height}). Returning original image.")
        return [img]

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
    # PSM 6: Assume a single uniform block of text. Often good for isolated coupons.
    # PSM 3: Fully automatic page segmentation (default). Good for general documents.
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    try:
        text = pytesseract.image_to_string(img, config=config).strip()
        
        # Common OCR corrections
        ocr_corrections = {
            '|': 'I', '0': 'O', 'vv': 'W', '1': 'I', '5': 'S',
            'S ': '$', # Correct common OCR error for dollar sign
            'S.': '$', # Correct common OCR error for dollar sign
            's ': '$', # Correct common OCR error for dollar sign
            's.': '$', # Correct common OCR error for dollar sign
            'OFFF': 'OFF', # Common OCR error
            'OF F': 'OFF', # Common OCR error
            'O FF': 'OFF', # Common OCR error
            'OFE': 'OFF', # Common OCR error
            'LIMITS': 'LIMIT 5', # Common OCR error
            'LIMITO': 'LIMIT 0', # Common OCR error (though unlikely for actual limits)
            'LIMI T': 'LIMIT',
            'WHILE SUPPLIES LAST': 'While Supplies Last', # Normalize casing
            'IN-WAREHOUSE': 'In-Warehouse',
            'ONLINE': 'Online',
            'WAREHOUSE': 'In-Warehouse',
            'ONLINEONLY': 'Online',
            'WAREHOUSEONLY': 'In-Warehouse'
        }
        for wrong, right in ocr_corrections.items():
            text = text.replace(wrong, right)
        
        # Replace newlines/tabs with spaces and normalize multiple spaces
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-alphanumeric characters except for known symbols like $, ., -, / and spaces
        text = re.sub(r'[^a-zA-Z0-9\s$/.,-]', '', text)
        
        logging.info(f"Extracted text from image (first 100 chars): {text[:100]}...")
        return text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def call_llm(prompt, model="mistral"):
    """
    Calls a local LLM using Ollama CLI to get a structured response.
    Ensure Ollama is running and the specified model is pulled.
    """
    try:
        logging.debug(f"Calling LLM with prompt (first 200 chars): {prompt[:200]}...")
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
        logging.debug(f"Raw LLM response: {output}")
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
    """
    Uses the LLM to extract structured item information from OCR text.
    Passes article-level metadata to the LLM for better context.
    """
    deal_type = "Hot Buy" if is_hot_buy else "Coupon Book"
    known_brands_str = ", ".join(known_brands)

    # Constructing a detailed prompt for the LLM
    prompt = f"""
You are an expert data extractor for Costco deals. The following text is raw OCR output from a coupon image. It may contain errors (e.g., "Ch4rm1n" instead of "Charmin", "T0il3t P4p3r" instead of "Toilet Paper").

Your task is to clean up the text and extract structured deal data for each item found.

Return a JSON list of dictionaries, where each dictionary represents one item and has the following fields:
- `item_brand`: The brand name of the product. **CRITICAL: You MUST select the brand name EXACTLY as it appears in the provided `Known Brands` list if a match is found, even if the OCR text has slight variations.** If no brand is explicitly mentioned or clearly inferable from the known brands, leave it empty.
- `item_description`: A clean and concise description of the item, fixing any OCR errors. **Form a coherent product description that includes key product attributes (e.g., size, quantity, features) if present in the text, similar to "Charmin Ultra Soft bath Tissue, 2-Ply, 213 Sheets, 30 Rolls".** Max 100 characters.
- `discount`: The exact discount wording as it appears (e.g., "$5 OFF", "20% OFF", "SAVE $10").
- `discount_cleaned`: The numerical value of the discount (e.g., "5", "20", "10"). Extract only the number. If not applicable, leave empty.
- `count_limit`: Any purchase limit (e.g., "Limit 2", "While Supplies Last"). If not specified, leave empty.
- `channel`: Where the deal is available (e.g., "In-Warehouse", "Online", "In-Warehouse + Online", "Warehouse-Only", "Book with Costco"). If not specified, leave empty.
- `discount_period`: The period during which the discount is valid. Extract this from the text if available. If not, infer from the article or leave empty.
- `item_original_price`: The original price of the item, if mentioned (e.g., "$119.99"). Extract only the price including the dollar sign. If not available, leave it empty.
- `source_url`: The URL of the image from which this data was extracted. (Set to "{source_url}" for all items extracted from this image).
- `article_name`: The name of the article this coupon belongs to. (Set to "{article_name}" for all items extracted from this image).
- `publish_date`: The publish date of the article. (Set to "{publish_date_str}" for all items extracted from this image).
- `type`: The type of deal. (Set to "{deal_type}" for all items extracted from this image).

**Known Brands for reference:** {known_brands_str}

**Important Rules:**
- Ensure `item_brand` is one of the `known_brands` if a match is found. If the OCR text has a highly confident match (e.g., "Dyson" for "Dyson Cyclone"), use the more specific known brand.
- If a field is missing or cannot be confidently extracted, return an an empty string for that field.
- Exclude any non-deal entries, headers, footers, or unrelated text from the output.
- Return ONLY the JSON list, no additional text, explanations, or markdown formatting outside the JSON.

---
OCR Text to Parse:
{raw_text}
---
"""
    response = call_llm(prompt)
    
    # Attempt to find and parse JSON from the LLM response
    try:
        # Find the first '[' and last ']' to isolate the JSON array
        json_start = response.find('[')
        json_end = response.rfind(']')
        
        if json_start == -1 or json_end == -1:
            logging.error("No complete JSON list found in LLM response.")
            logging.debug(f"Raw LLM response: {response}")
            return []
            
        json_string = response[json_start : json_end + 1]
        json_data = json.loads(json_string)
        
        # Post-processing for item_brand to ensure it's from known_brands
        for item in json_data:
            if item.get('item_brand'):
                # Use rapidfuzz for fuzzy matching with a high score cutoff
                match_tuple = process.extractOne(item['item_brand'], known_brands, score_cutoff=85)
                if match_tuple:
                    item['item_brand'] = match_tuple[0] # Use the exact known brand name
                else:
                    logging.debug(f"LLM extracted brand '{item['item_brand']}' did not high-confidently match known brands. Keeping original.")

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

def find_all_coupon_links(base_url="https://www.costcoinsider.com/category/coupons/", history_years=2):
    """
    Finds all coupon book and hot buys links from the main category page
    within the specified history_years.
    """
    logging.info(f"Starting to find all coupon links from {base_url} for the last {history_years} years.")
    all_links = []
    current_year = datetime.now().year
    
    current_category_page_url = base_url
    page_num = 1 

    while True:
        logging.info(f"Fetching category page: {current_category_page_url} (Logical Page: {page_num})")
        html = get_page(current_category_page_url)
        if not html:
            logging.warning(f"Could not fetch {current_category_page_url}. Stopping link discovery.")
            break
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # --- NEW: More robust selection for article links ---
        # Look for links within h2, h3, or any 'a' tag that has 'costco-' in its href
        # and is likely part of a post title or content.
        potential_article_links = soup.select('h2.entry-title a, h3.entry-title a, div.entry-content a[href*="costco-"], div.post-content a[href*="costco-"]')
        logging.debug(f"Found {len(potential_article_links)} potential article links directly.")

        # --- DIAGNOSTIC: Print HTML snippet if no links found ---
        if not potential_article_links:
            logging.warning(f"No potential article links found with current selectors. Dumping first 2000 chars of HTML:")
            logging.warning(html[:2000])


        articles_found_on_page_for_filter = False # Flag to check if any *relevant* articles were found

        for a_tag in potential_article_links:
            title = a_tag.text.strip()
            href = a_tag.get('href')

            if not href or not title:
                logging.debug(f"Skipping link with no href or title: {title} - {href}")
                continue

            href = urljoin(base_url, href)
            
            publish_date_str = "Unknown Date"
            article_year = current_year
            
            # Try to find the closest parent element that might contain a 'div.postdate'
            # This is more flexible than assuming 'article' as the direct parent.
            parent_with_date = a_tag.find_parent(lambda tag: tag.name in ['article', 'div'] and tag.select_one('div.postdate'))
            if parent_with_date:
                publish_date_tag = parent_with_date.select_one('div.postdate')
                if publish_date_tag:
                    date_text = publish_date_tag.text.strip()
                    try:
                        parsed_date = datetime.strptime(date_text, '%B %d, %Y')
                        publish_date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                        article_year = parsed_date.year
                    except ValueError:
                        logging.warning(f"Could not parse publish date '{date_text}' for '{title}'.")
            
            # Fallback for year extraction if date parsing failed or div.postdate was missing
            if publish_date_str == "Unknown Date" or (article_year == current_year and not re.search(r'\b(20\d{2})\b', title + href)):
                year_match = re.search(r'\b(20\d{2})\b', title + href)
                if year_match:
                    article_year = int(year_match.group(1))

            if current_year - article_year > history_years:
                logging.info(f"Skipping old article: {title} ({article_year}). Reached historical limit of {history_years} years.")
                # Assuming articles are generally ordered from newest to oldest on category pages
                return all_links 
            
            # Filter for coupon book or hot buys links based on title or href keywords
            if 'hot buys' in title.lower() or 'coupon book' in title.lower() or \
               'hot-buys' in href.lower() or 'coupon-book' in href.lower():
                
                if href not in [link_info['url'] for link_info in all_links]:
                    all_links.append({
                        'title': title,
                        'url': href,
                        'is_hot_buy': 'hot buys' in title.lower() or 'hot-buys' in href.lower(),
                        'publish_date': publish_date_str 
                    })
                    articles_found_on_page_for_filter = True # Mark that we found at least one relevant article
                    logging.info(f"Found relevant link: {title} - {href} (Published: {publish_date_str})")
                else:
                    logging.debug(f"Skipping duplicate relevant link: {href}")
            else:
                logging.debug(f"Skipping irrelevant link (no 'coupon book' or 'hot buys' in title/href): {title} - {href}")
        
        # If no relevant articles were found on the current page, it means we've likely hit the end of content
        if not articles_found_on_page_for_filter and page_num > 1: # Only stop if it's not the very first page and nothing was found
            logging.info(f"No relevant coupon/hot buy articles found on page {page_num}. Stopping link discovery.")
            break
        elif not articles_found_on_page_for_filter and page_num == 1:
            logging.warning(f"No relevant articles found on the first page ({current_category_page_url}). This might indicate a selector issue or no content.")
            # Don't break immediately on page 1 if nothing found, allow to check pagination
            
        # Find the "Older Entries" link for pagination
        older_entries_div = soup.select_one(CONFIG['pagination_selectors']['category_older'])
        next_page_link = older_entries_div.find('a') if older_entries_div else None

        # Ensure the link text is "Older Entries" to avoid misidentifying pagination
        if next_page_link and next_page_link.get_text(strip=True) == '« Older Entries':
            current_category_page_url = next_page_link['href']
            page_num += 1
            logging.info(f"Moving to next category page: {current_category_page_url}")
            time.sleep(CONFIG['delay']) 
        else:
            logging.info("No more valid pagination links ('« Older Entries') found. Stopping link discovery.")
            break

    return all_links

def scrape_images_from_page(url, article_name, publish_date_str, is_hot_buy=False, visited_urls=None, current_article_page_count=0):
    """
    Scrapes images from a given article page, processes them with OCR and LLM,
    and returns a list of structured item data. Handles internal article pagination.
    """
    if visited_urls is None:
        visited_urls = set()
    
    # Check max pages for *this specific article*
    if current_article_page_count >= CONFIG['max_article_pages']:
        logging.info(f"Reached maximum page limit ({CONFIG['max_article_pages']}) for article: {article_name}. Stopping internal pagination.")
        return []
    
    if url in visited_urls:
        logging.info(f"Already visited {url} within this article. Skipping to avoid loop.")
        return []
    
    visited_urls.add(url)
    current_article_page_count += 1
    
    logging.info(f"Scraping images from article page: {url} (Article Page {current_article_page_count}/{CONFIG['max_article_pages']})")
    html = get_page(url)
    if not html:
        return []
    
    # Load brands based on type (hot buy or coupon book)
    brand_file = 'hot_buy_brands.txt' if is_hot_buy else 'coupon_book_brands.txt'
    known_brands = load_known_brands(brand_file)
    
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    
    # Select images based on configured selectors
    selectors = CONFIG['image_selectors']['hot_buy'] if is_hot_buy else CONFIG['image_selectors']['coupon']
    page_images = []
    for selector in selectors:
        found_images = soup.select(selector)
        if found_images:
            page_images.extend(found_images)
            # Heuristic: If we found images with a specific selector, prioritize them
            # This helps avoid grabbing all random images if a more precise selector worked.
            # Removed the specific check for '#coupon-book' as it's not always reliable
            if len(found_images) > 0 and ('coupon' in selector or 'hotbuy' in selector or 'deal' in selector):
                break # Stop after finding relevant images with a good selector
    
    # Fallback if specific selectors don't yield results
    if not page_images:
        page_images = soup.select('img')
        logging.warning(f"No specific images found for {url}, falling back to all <img> tags. Found {len(page_images)}.")
    else:
        # Remove duplicates if selectors overlapped
        page_images = list(set(page_images))
        logging.info(f"Final count of unique images found for {url}: {len(page_images)}.")

    for img_tag in page_images:
        img_url = img_tag.get('src') or img_tag.get('data-src') # Check data-src as well
        if not img_url:
            logging.debug(f"Skipping image tag with no src or data-src attribute in {url}")
            continue
            
        # Resolve relative URLs
        img_url = urljoin(url, img_url)
        
        # Extract the filename from the URL for exclusion checks
        parsed_img_url = urlparse(img_url)
        filename = os.path.basename(parsed_img_url.path).lower()
        
        # Skip excluded images (e.g., logos, banners)
        if any(exclude.lower() in filename for exclude in CONFIG['exclude_images']):
            logging.info(f"Skipping excluded image: {img_url} (matched pattern in filename: {filename})")
            continue
            
        logging.info(f"Attempting to download and process image: {img_url}")
        img = download_image(img_url, url)
        if not img:
            logging.warning(f"Failed to download or preprocess image: {img_url}")
            continue
        
        # --- Image Splitting Logic (Applied to all large images) ---
        # Apply splitting if the image is large enough, regardless of hot buy/coupon book type.
        # The split_grid_image function's rows/cols will be determined here.
        if img.width > 500 and img.height > 500: # Heuristic for potentially multi-item images
            # For hot buys, assume 2 rows and 4 columns based on common layouts
            # For coupon books, use a more general 2 rows x 2 columns split as a starting point
            rows_to_split = 2
            cols_to_split = 4 if is_hot_buy else 2 
            
            sub_images = split_grid_image(img, rows=rows_to_split, cols=cols_to_split)
            logging.info(f"Image split into {len(sub_images)} sub-images for OCR (Rows: {rows_to_split}, Cols: {cols_to_split}).")
        else:
            # For smaller or non-grid images, process the entire image as one block.
            sub_images = [img] 
            logging.info(f"Processing image as a single block (not split due to size or type).")

        for sub_img in sub_images:
            text = extract_text_from_image(sub_img)
            # Minimum text length to avoid processing empty or very noisy OCR results
            if not text or len(text.strip()) < 20: 
                logging.debug(f"Skipping sub-image with insufficient text (length {len(text.strip())}): {text[:50]}...")
                continue
            
            # Use LLM to parse the extracted text
            parsed_items = parse_coupon_data(text, img_url, article_name, publish_date_str, is_hot_buy, known_brands)
            if parsed_items:
                items.extend(parsed_items)
            else:
                logging.warning(f"LLM returned no parsed items for text block: {text[:100]}...")
    
    # Handle internal article pagination (e.g., for multi-page hot buys articles)
    # The selector here should be for pagination *within* an article, not category pages
    pagination_selectors = CONFIG['pagination_selectors']['article_next_page'].split(', ')
    next_article_page_link = None
    for selector in pagination_selectors:
        next_article_page_link = soup.select_one(selector)
        if next_article_page_link and next_article_page_link.get('href'):
            # Ensure it's not a link back to the category page or a duplicate of the current page
            # Also, check if the link text suggests it's a "next" page
            link_text = next_article_page_link.get_text().lower()
            if "next" in link_text or "›" in link_text or ">" in link_text:
                if urlparse(next_article_page_link['href']).path != urlparse(url).path:
                    break # Found a valid next page link within the article
                else:
                    next_article_page_link = None # It's a link to the current page or similar, ignore
            else:
                next_article_page_link = None # Not a "next" link, ignore

    if next_article_page_link and next_article_page_link.get('href'):
        next_url = urljoin(url, next_article_page_link['href'])
        logging.info(f"Following next article page: {next_url}")
        # Recursively call to scrape next page, passing visited_urls and incremented page_count
        items.extend(scrape_images_from_page(next_url, article_name, publish_date_str, is_hot_buy, visited_urls, current_article_page_count))
    else:
        logging.info(f"No more internal article pagination links found for {url}.")
    
    return items

def write_to_csv(items):
    """
    Writes extracted items to separate CSV files based on their 'type' (Coupon Book or Hot Buy).
    Handles appending to existing files and writing headers only once.
    """
    # Define the final fieldnames for the CSV, matching the LLM output (or mapping them)
    fieldnames = [
        "item_brand", "item_description", "discount", "discount_cleaned",
        "count_limit", "channel", "discount_period", "item_original_price",
        "source_url", "article_name", "publish_date", "type"
    ]
    
    coupon_items = [item for item in items if item.get('type') == "Coupon Book"]
    hot_buy_items = [item for item in items if item.get('type') == "Hot Buy"] # Assuming "Hot Buy" covers all hot buys

    # Write Coupon Book items
    if coupon_items:
        file_exists = os.path.isfile(COUPON_OUTPUT_FILE)
        with open(COUPON_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in coupon_items:
                # Ensure all fields are present, even if empty, to avoid KeyError
                row_data = {field: item.get(field, "") for field in fieldnames}
                writer.writerow(row_data)
        logging.info(f"Saved {len(coupon_items)} coupon book items to {COUPON_OUTPUT_FILE}")

    # Write Hot Buy items
    if hot_buy_items:
        file_exists = os.path.isfile(HOT_BUYS_OUTPUT_FILE)
        with open(HOT_BUYS_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in hot_buy_items:
                # Ensure all fields are present, even if empty, to avoid KeyError
                row_data = {field: item.get(field, "") for field in fieldnames}
                writer.writerow(row_data)
        logging.info(f"Saved {len(hot_buy_items)} hot buy items to {HOT_BUYS_OUTPUT_FILE}")

def main():
    """Main function to orchestrate the scraping process."""
    if not initialize(): 
        logging.error("Scraper initialization failed. Exiting.")
        return

    logging.info("Starting comprehensive Costco coupon scraping.")
    
    # Use find_all_coupon_links to get all relevant articles for the last 2 years
    all_post_links = find_all_coupon_links(history_years=CONFIG['history_years'])
    
    if not all_post_links:
        logging.error("No valid coupon or hot buy post links found. Exiting.")
        return

    total_items_scraped = []
    for post_info in all_post_links:
        title = post_info['title']
        link = post_info['url']
        is_hot_buy = post_info['is_hot_buy']
        publish_date_str = post_info['publish_date']
        
        logging.info(f"\n--- Processing: {'Hot Buys' if is_hot_buy else 'Coupon Book'} - {title} (Published: {publish_date_str}) ---")
        
        # Scrape images and extract data for the current article, including internal pagination
        items_from_article = scrape_images_from_page(link, title, publish_date_str, is_hot_buy)
        
        if items_from_article:
            total_items_scraped.extend(items_from_article)
            logging.info(f"Collected {len(items_from_article)} items from article: {title}")
        else:
            logging.warning(f"No items extracted from article: {title}")

    # Write all collected items to CSV files at the end
    if total_items_scraped:
        write_to_csv(total_items_scraped)
        logging.info(f"🎉 Done! Total {len(total_items_scraped)} items scraped and saved to CSV files.")
    else:
        logging.warning("No items were extracted from any articles during the entire run.")

if __name__ == "__main__":
    main()
