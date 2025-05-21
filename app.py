# Costco Coupon + Hot Buys LLM-Based Scraper System

import requests
from bs4 import BeautifulSoup
import json
import csv
import os
from datetime import datetime

BASE_URL = "https://www.costcoinsider.com/category/coupons/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_FILE = "costco_deals_output.csv"


def get_monthly_links():
    """Scrape the main coupons category page for monthly coupon/hot buys links."""
    res = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")
    links = []
    for a in soup.select("article a"):
        href = a.get("href")
        if href and "coupon-book" in href:
            links.append(href)
        if href and "hot-buys-coupons" in href:
            links.append(href)
    return list(set(links))  # deduplicate


def get_page_text(url):
    """Scrape and clean the main content text from a page."""
    try:
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")
        content = soup.select_one(".entry-content")
        return content.get_text(separator="\n") if content else ""
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""


def call_llm(prompt, model="mistral"):
    """Call local LLM using Ollama CLI (ensure model is pulled)."""
    import subprocess
    try:
        result = subprocess.run([
            "ollama", "run", model
        ], input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return result.stdout.decode()
    except Exception as e:
        print("LLM call failed:", e)
        return ""


def extract_items_from_text(raw_text):
    """Use the LLM to extract structured item info from text."""
    prompt = f"""
Extract structured deal data from the Costco deals text below. Return only JSON list with these fields:
item_name, description, discount_value, channel, validity_period, item_limit, type

---
{raw_text}
---
"""
    response = call_llm(prompt)
    try:
        # Try to isolate JSON part
        json_start = response.find('[')
        json_data = json.loads(response[json_start:])
        return json_data
    except Exception as e:
        print("JSON parsing failed:", e)
        return []


def write_to_csv(items, file_path=OUTPUT_FILE):
    """Append items to CSV."""
    fieldnames = ["item_name", "description", "discount_value", "channel", "validity_period", "item_limit", "type"]
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for item in items:
            writer.writerow(item)


def run_pipeline():
    links = get_monthly_links()
    print(f"Found {len(links)} monthly links.")
    for link in links:
        print(f"Processing: {link}")
        page_text = get_page_text(link)
        if len(page_text) < 100:
            continue
        items = extract_items_from_text(page_text)
        if items:
            write_to_csv(items)
            print(f"Added {len(items)} items from {link}")


if __name__ == "__main__":
    run_pipeline()
