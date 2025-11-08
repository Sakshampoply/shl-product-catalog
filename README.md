# SHL Assessment Recommendation System# SHL Product Catalog Scraper

An intelligent recommendation system for SHL assessments using **Multi-Vector Retrieval** with Query Deconstruction and LLM Reranking.Scrapes the "Individual Test Solutions" table from:

## 🎯 Performance- https://www.shl.com/products/product-catalog/

- **Multi-Vector V2**: **39.11% Mean Recall@10**It saves JSON and CSV with columns:

- Query Expansion: 28.67%

- Gemini Embeddings: 25.89%- name

- Baseline (sentence-transformers): 21.33%- url

- remote_testing (Yes/No/empty)

## 🏗️ Architecture- adaptive_iri (Yes/No/empty)

- test_type_keys (space-separated letter codes)

### Multi-Vector Retrieval V2

## How to run

The system uses a sophisticated 3-stage pipeline:

````bash

1. **Query Deconstruction** (Gemini 2.5 Flash Lite)# Optional: create a virtual environment

   - Breaks complex queries into focused search facetspython3 -m venv .venv

   - Example: "Java developers who collaborate" → source .venv/bin/activate

     - "Java programming technical skills"

     - "collaboration communication skills"# Install dependencies

     - "40 minute assessment time"pip install -r requirements.txt



2. **Multi-Vector Semantic Search**# Run the scraper (defaults to the URL above)

   - Searches independently for each facetpython scrape_shl_catalog.py

   - Uses pre-computed Gemini embeddings (3072-dim)

   - Merges candidates from all facets# Or scrape from a specific page (if needed)

python scrape_shl_catalog.py "https://www.shl.com/products/product-catalog/"

3. **LLM Reranking** (Gemini 2.5 Flash Lite)```

   - Ensures balanced recommendations (technical + behavioral)

   - Reranks based on query-assessment fitOutputs are written to `output/shl_catalog.json` and `output/shl_catalog.csv`.

   - Returns top-10 assessments

## Notes

## 🚀 Quick Start

- The script uses requests + BeautifulSoup; no browser automation required. If the site’s HTML changes, the selectors try to be resilient but may need minor tweaks.

### Installation- Pagination: it follows the nearest "Next" link to the table and aggregates rows until there is no next page.

- Please respect the website’s terms of use and robots directives. Add delays if you widen the crawl.

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env
````

### API Usage

```bash
# Start the API server
python api.py
# or
uvicorn api:app --reload

# Visit http://localhost:8000/docs for interactive API documentation
```

### Evaluation

```bash
# Evaluate on training set
export $(cat .env | xargs) && python evaluate.py

# Expected output: ~39.11% Mean Recall@10
```

### Generate Test Predictions

```bash
# Generate predictions for test dataset
export $(cat .env | xargs) && python generate_predictions.py

# Output: test_predictions.csv
```

## 📁 Project Structure

```
SHL/
├── api.py                              # FastAPI backend
├── evaluate.py                         # Training set evaluation
├── generate_predictions.py             # Test set prediction generator
├── evaluate_multi_vector_v2.py        # Multi-vector evaluation script
├── src/
│   ├── multi_vector_retriever_v2.py   # Main retriever (3-stage pipeline)
│   ├── query_deconstructor_v2.py      # Query → facets (Gemini)
│   ├── llm_reranker_v2.py             # LLM-based reranking
│   └── evaluator.py                   # Recall@k metrics
├── data/
│   └── gemini/
│       ├── faiss_gemini.index         # FAISS index (353 assessments)
│       ├── embeddings_gemini.npy      # Pre-computed embeddings
│       └── metadata_gemini.json       # Assessment metadata
└── output/
    ├── shl_catalog.json               # Scraped catalog data
    └── shl_catalog.csv
```

## 🔧 Data Scraping

The scraper collects SHL's "Individual Test Solutions" from:
https://www.shl.com/products/product-catalog/

```bash
python scrape_shl_catalog.py
```

Outputs: `output/shl_catalog.json` and `output/shl_catalog.csv`

## 📊 Evaluation Results

**Multi-Vector V2 Performance** (Training Set, n=10 queries):

| Query                 | Recall@10  |
| --------------------- | ---------- |
| Java + Collaboration  | 60.00%     |
| Sales (Entry-level)   | 33.33%     |
| COO (China)           | 50.00%     |
| Radio Station Manager | 40.00%     |
| Content Writer        | 60.00%     |
| QA Engineer           | 77.78%     |
| Bank Assistant        | 0.00%      |
| Marketing Manager     | 20.00%     |
| Consultant            | 20.00%     |
| Senior Data Analyst   | 30.00%     |
| **Mean**              | **39.11%** |

### Why Multi-Vector Works

**Problem Solved**: Queries often need BOTH technical AND behavioral assessments. Single-vector search may bias toward one type.

**Solution**:

- Deconstruct "Java developer who collaborates" into separate facets
- Search independently for Java skills AND collaboration skills
- LLM ensures balanced recommendations

**Result**: 36% relative improvement over previous best (28.67% → 39.11%)

## 🛠️ API Endpoints

### `POST /recommend`

Get assessment recommendations for a query.

**Request:**

```json
{
  "query": "I need Java developers who can collaborate effectively",
  "top_k": 10
}
```

**Response:**

```json
{
  "query": "I need Java developers who can collaborate effectively",
  "recommendations": [
    {
      "name": "Java Programming Assessment",
      "url": "https://www.shl.com/...",
      "description": "...",
      "test_types": ["K", "P"],
      "duration_minutes": 45,
      "job_levels": ["Entry", "Mid"],
      "score": 0.89
    }
  ],
  "total": 10
}
```

### `GET /health`

Health check endpoint.

## 📝 Notes

- Uses Gemini 2.5 Flash Lite for text generation (query deconstruction, reranking)
- Uses embedding-001 for semantic search (pre-computed embeddings)
- Respects API rate limits with exponential backoff
- Free tier quotas: 100 RPM embeddings, 10 RPM text generation

## 🔒 Environment Variables

Required in `.env`:

```bash
GOOGLE_API_KEY=your_api_key_here
```

## 📄 License

This project is for educational purposes. Please respect SHL's website terms of use when scraping.
