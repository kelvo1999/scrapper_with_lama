import requests
from bs4 import BeautifulSoup
import json
import csv
import os
from datetime import datetime
import logging


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
BASE_URL = "https://www.costcoinsider.com/category/coupons/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
COUPON_OUTPUT_FILE = "coupon_books.csv"
HOT_BUYS_OUTPUT_FILE = "hot_buys.csv"

def get_monthly_links():
    """Scrape the main coupons category page for monthly coupon/hot buys links."""
    try:
        res = requests.get(BASE_URL, headers=HEADERS)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        links = {'coupon': [], 'hot_buy': []}
        for a in soup.select("article a"):
            href = a.get("href")
            if not href:
                continue
            if "coupon-book" in href:
                links['coupon'].append(href)
                logging.info(f"Found coupon book link: {href}")
            elif "hot-buys-coupons" in href:
                links['hot_buy'].append(href)
                logging.info(f"Found hot buys link: {href}")
        # Deduplicate
        links['coupon'] = list(set(links['coupon']))
        links['hot_buy'] = list(set(links['hot_buy']))
        return links
    except Exception as e:
        logging.error(f"Failed to fetch links from {BASE_URL}: {e}")
        return {'coupon': [], 'hot_buy': []}

def get_page_text(url):
    """Scrape and clean the main content text from a page."""
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        content = soup.select_one(".entry-content")
        text = content.get_text(separator="\n") if content else ""
        logging.info(f"Extracted {len(text)} characters from {url}")
        return text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return ""

def call_llm(prompt, model="mistral"):
    """Call local LLM using Ollama CLI (ensure model is pulled)."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60  # Add timeout to prevent hanging
        )
        output = result.stdout.decode().strip()
        if result.stderr:
            logging.warning(f"LLM stderr: {result.stderr.decode()}")
        logging.info(f"LLM response length: {len(output)} characters")
        return output
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return ""

def extract_items_from_text(raw_text, is_hot_buy):
    """Use the LLM to extract structured item info from text."""
    deal_type = "Hot Buy" if is_hot_buy else "Coupon Book"
    prompt = f"""
You are an expert data extractor for Costco deals. Extract structured deal data from the text below. Return a JSON list with these fields for each item:
- item_name (product name, clean and concise, max 50 characters)
- description (brief item description, max 100 characters)
- discount_value (discount amount, e.g., "$5 OFF" or "$10", clean format)
- channel (e.g., "In-Warehouse", "Online", "In-Warehouse + Online", "Warehouse-Only", or "Book with Costco")
- validity_period (date range, e.g., "3/29/25 - 4/6/25", or empty if not specified)
- item_limit (e.g., "Limit 2" or "While Supplies Last", or empty if not specified)
- type (set to "{deal_type}" for all items)

Rules:
- Ensure item_name and description are clean, removing typos and special characters.
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
        # Try to isolate JSON part
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

def write_to_csv(items):
    """Write items to separate CSV files based on type."""
    fieldnames = ["item_name", "description", "discount_value", "channel", "validity_period", "item_limit", "type"]
    
    # Separate items into coupon books and hot buys
    coupon_items = [item for item in items if item.get('type') == "Coupon Book"]
    hot_buy_items = [item for item in items if item.get('type') in ["Hot Buy", "Travel Hot Buy"]]

    # Write Coupon Book items
    if coupon_items:
        file_exists = os.path.isfile(COUPON_OUTPUT_FILE)
        with open(COUPON_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in coupon_items:
                writer.writerow(item)
        logging.info(f"Saved {len(coupon_items)} coupon book items to {COUPON_OUTPUT_FILE}")

    # Write Hot Buy items
    if hot_buy_items:
        file_exists = os.path.isfile(HOT_BUYS_OUTPUT_FILE)
        with open(HOT_BUYS_OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in hot_buy_items:
                writer.writerow(item)
        logging.info(f"Saved {len(hot_buy_items)} hot buy items to {HOT_BUYS_OUTPUT_FILE}")

def run_pipeline():
    links = get_monthly_links()
    total_links = len(links['coupon']) + len(links['hot_buy'])
    logging.info(f"Found {total_links} monthly links: {len(links['coupon'])} coupon, {len(links['hot_buy'])} hot buy")

    # Process Coupon Book links
    for link in links['coupon']:
        logging.info(f"Processing Coupon Book: {link}")
        page_text = get_page_text(link)
        if len(page_text) < 100:
            logging.warning(f"Page text too short for {link}, skipping")
            continue
        items = extract_items_from_text(page_text, is_hot_buy=False)
        if items:
            write_to_csv(items)
            logging.info(f"Added {len(items)} items from {link}")
        else:
            logging.warning(f"No items extracted from {link}")

    # Process Hot Buy links
    for link in links['hot_buy']:
        logging.info(f"Processing Hot Buy: {link}")
        page_text = get_page_text(link)
        if len(page_text) < 100:
            logging.warning(f"Page text too short for {link}, skipping")
            continue
        items = extract_items_from_text(page_text, is_hot_buy=True)
        if items:
            write_to_csv(items)
            logging.info(f"Added {len(items)} items from {link}")
        else:
            logging.warning(f"No items extracted from {link}")

if __name__ == "__main__":
    run_pipeline()