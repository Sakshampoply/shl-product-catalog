# SHL Product Catalog Scraper

Scrapes the "Individual Test Solutions" table from:

- https://www.shl.com/products/product-catalog/

It saves JSON and CSV with columns:

- name
- url
- remote_testing (Yes/No/empty)
- adaptive_iri (Yes/No/empty)
- test_type_keys (space-separated letter codes)

## How to run

```bash
# Optional: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the scraper (defaults to the URL above)
python scrape_shl_catalog.py

# Or scrape from a specific page (if needed)
python scrape_shl_catalog.py "https://www.shl.com/products/product-catalog/"
```

Outputs are written to `output/shl_catalog.json` and `output/shl_catalog.csv`.

## Notes

- The script uses requests + BeautifulSoup; no browser automation required. If the site’s HTML changes, the selectors try to be resilient but may need minor tweaks.
- Pagination: it follows the nearest "Next" link to the table and aggregates rows until there is no next page.
- Please respect the website’s terms of use and robots directives. Add delays if you widen the crawl.
