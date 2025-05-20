✅ What This Script Does
Scrapes the CostcoInsider coupon and hot buy URLs monthly from the category/coupons index.

Fetches page content and cleans it into readable text.

Sends content to a local LLM (via Ollama, e.g., mistral, llama3, etc.) to extract structured deal data.

Writes extracted deals into a CSV file with all necessary fields.

⚙️ Requirements to Run
Python packages:

bash
pip install beautifulsoup4 requests
Ollama setup (for LLM inference):

Install Ollama

Pull a model (e.g.):

bash
Copy
Edit
ollama pull mistral
You can replace mistral with any other installed model in the script.

📦 CSV Output Format
item_name	description	discount_value	channel	validity_period	item_limit	type

