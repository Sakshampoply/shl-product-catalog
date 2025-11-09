# SHL Assessment Recommendation System

An intelligent AI-powered recommendation system that helps recruiters find the most relevant SHL assessments for their hiring needs using Multi-Vector Retrieval, Query Deconstruction, and LLM Reranking.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Performance](#performance)
- [Local Setup](#local-setup)
- [Project Structure](#project-structure)
- [Approach Multi-Vector Retrieval](#approach-multi-vector-retrieval)
- [Evaluation Metrics](#evaluation-metrics)
- [API Documentation](#api-documentation)
- [Frontend](#frontend)
- [Usage](#usage)

---

## Overview

This system solves the challenge of matching complex recruiter queries to SHL's catalog of 353 assessments. Unlike simple keyword matching, it understands nuanced requirements.

**Example Queries:**

- "I need Java developers who can also collaborate effectively" → Recommends both technical (Java) and behavioral (collaboration) tests
- "Senior data analyst with SQL and Python" → Considers job level, technical skills, and problem-solving abilities
- "40-minute assessment for QA engineer" → Factors in duration constraints and role-specific competencies

**Key Features:**

- Query understanding via LLM-based deconstruction
- Balanced recommendations (technical plus behavioral tests)
- Fast semantic search using FAISS
- Production-ready REST API
- Clean SHL-branded web interface

---

## Tech Stack

### Backend

- **Python 3.11** - Core language
- **FastAPI** - REST API framework with automatic OpenAPI docs
- **FAISS** - Facebook's vector similarity search library
- **Google Gemini AI** - LLM for query understanding and reranking
  - gemini-2.5-flash-lite - Query deconstruction and reranking
  - gemini-embedding-001 - 3072-dimensional embeddings
- **NumPy** - Efficient array operations
- **Pandas** - Data manipulation and CSV handling
- **BeautifulSoup4** - Web scraping for catalog data
- **Uvicorn** - ASGI server

### Frontend

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first styling with SHL branding
- **React Hooks** - State management
- **SHL Design System** - Green (hex 78D64B) and gray (hex 4a4a4a) color scheme

### Development Tools

- **Jupyter Notebook** - Interactive evaluation and predictions
- **Git** - Version control

---

## Performance

### Multi-Vector Retrieval System

**Mean Recall at 10: 38 to 68 percent (Typical average 50 to 55 percent)**

**IMPORTANT NOTE ON NON-DETERMINISM**

Due to the LLM-based components (query deconstruction and reranking), the system's performance is NOT deterministic. The same query may produce slightly different results across runs.

**Observed Recall at 10 Range: 0.38 to 0.68**

**Why Performance Varies:**

- LLM temperature and sampling randomness
- Different query deconstructions across runs
- Varying reranking decisions
- Typical variance: plus or minus 10 to 15 percent between runs

**Performance Characteristics:**

- **Best Performing:** Multi-faceted queries (technical plus behavioral)
  - Example: "Java developer who collaborates" → 40 to 70 percent
  - Example: "QA Engineer" → 60 to 80 percent
- **Moderate:** Single-skill technical queries → 30 to 50 percent
- **Challenging:** Very specific niche roles → 0 to 30 percent

---

## Local Setup

### Prerequisites

- **Python 3.11 or higher** - Backend runtime
- **Node.js 18 or higher** - Frontend runtime
- **Google Gemini API Key** - Get one free at https://aistudio.google.com/app/apikey
- **Jupyter** (optional) - For running evaluation notebook

### Step 1: Clone Repository

```bash
git clone https://github.com/Sakshampoply/shl-product-catalog.git
cd shl-product-catalog
```

### Step 2: Backend Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# On Windows use: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Step 3: Verify Data Files

Ensure the following data files exist (included in repo):

```
data/
├── catalog_records.json
└── gemini/
    ├── faiss_gemini.index
    ├── embeddings_gemini.npy
    └── metadata_gemini.json
```

### Step 4: Start Backend API

```bash
# Start FastAPI server
python api.py

# Or use uvicorn directly
uvicorn api:app --reload --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Step 5: Frontend Setup

The frontend provides a clean, SHL-branded interface for the recommendation system.

```bash
# Open new terminal
cd frontend

# Install Node.js dependencies
npm install

# Start Next.js development server
npm run dev

# Frontend available at http://localhost:3000
```

**Frontend Features:**

- SHL-branded design (green hex 78D64B and white color scheme)
- Interactive search with real-time results
- Top 10 recommendations with full metadata
- Responsive design (mobile-friendly)
- Fast client-side rendering with Next.js 14

### Step 6: Test the System

**Option A: Web Interface (Recommended)**

1. Open browser to http://localhost:3000
2. Enter a query: "I need Python developers who are good at problem solving"
3. Click Search
4. View top 10 recommended assessments with details

**Option B: API Testing**

Visit http://localhost:8000/docs for interactive Swagger UI

- Test the recommend endpoint
- View request and response schemas
- Try different queries

**Option C: Evaluation Notebook**

```bash
# Start Jupyter
jupyter notebook evaluation_and_predictions.ipynb

# Run cells to:
# 1. Evaluate on training data
# 2. Generate test predictions
# 3. Export results to CSV
```

---

## Project Structure

```
SHL/
├── api.py
├── scrape_shl_catalog.py
├── build_index.py
├── evaluation_and_predictions.ipynb
├── requirements.txt
├── .env
│
├── src/
│   ├── multi_vector_retriever_v2.py
│   ├── query_deconstructor_v2.py
│   ├── llm_reranker_v2.py
│   └── evaluator.py
│
├── data/
│   ├── catalog_records.json
│   ├── bm25_corpus.json
│   ├── metadata.json
│   └── gemini/
│       ├── faiss_gemini.index
│       ├── embeddings_gemini.npy
│       └── metadata_gemini.json
│
├── output/
│   ├── shl_catalog.json
│   └── shl_catalog.csv
│
└── frontend/
    ├── app/
    │   ├── page.tsx
    │   ├── layout.tsx
    │   └── globals.css
    ├── components/
    │   ├── SearchBar.tsx
    │   └── ResultsDisplay.tsx
    ├── public/
    │   └── shl-logo.svg
    ├── package.json
    ├── tailwind.config.ts
    └── tsconfig.json
```

### File Purposes

**Backend Files:**

- **api.py** - FastAPI server exposing recommend and health endpoints. Handles CORS, request validation, and response formatting.
- **scrape_shl_catalog.py** - Scrapes SHL's product catalog using BeautifulSoup. Extracts name, URL, description, test types, duration, job levels.
- **build_index.py** - Builds FAISS index from embeddings and metadata for efficient similarity search.
- **evaluation_and_predictions.ipynb** - Jupyter notebook for evaluating system performance on training data and generating test predictions. Replaces evaluate.py, generate_predictions.py, and evaluate_multi_vector_v2.py.

**Core Retrieval System (src directory):**

- **multi_vector_retriever_v2.py** - Orchestrates the 3-stage pipeline (deconstruction → search → reranking). Main entry point for retrieval.
- **query_deconstructor_v2.py** - Uses Gemini to break queries into focused facets (technical skills, behavioral traits, constraints).
- **llm_reranker_v2.py** - Uses Gemini to rerank candidates ensuring balanced recommendations (technical plus behavioral tests).
- **evaluator.py** - Computes Recall at k metrics comparing predictions to ground truth.

**Data Files (data directory):**

- **catalog_records.json** - 353 SHL assessments with full metadata (name, URL, description, test types, duration, job levels).
- **gemini/faiss_gemini.index** - FAISS index for cosine similarity search on 3072-dim Gemini embeddings.
- **gemini/embeddings_gemini.npy** - Pre-computed embeddings to avoid API quota issues at runtime.

**Frontend Files (frontend directory):**

- **app/page.tsx** - Main React component with search interface, results display, SHL header with logo, and footer.
- **components/SearchBar.tsx** - Controlled input with loading states, search icon, and green submit button.
- **components/ResultsDisplay.tsx** - Renders top 10 assessments with metadata (name, description, test types, duration, job levels, clickable URLs).

---

## Approach Multi-Vector Retrieval

### The Problem: Single-Vector Limitations

Traditional semantic search uses a single query embedding to find similar documents. This fails for complex queries requiring multiple distinct skill types.

**Example Query:** "I need Java developers who can also collaborate effectively"

**Single-Vector Issue:**

- Embeds entire query as one vector
- Search results biased toward either Java OR collaboration
- Misses assessments covering both skills
- Cannot balance multiple distinct requirements

### The Solution: Multi-Vector Retrieval with Query Deconstruction

Our system uses a 3-stage pipeline to ensure balanced, comprehensive recommendations.

---

### Stage 1: Query Deconstruction

**Goal:** Break complex queries into focused search facets

**How:** Use Gemini 2.5 Flash Lite to identify distinct skill domains

**Example:**

Input Query: "Java developers who collaborate, 40 min assessment"

Deconstructed Facets:

1. "Java programming technical skills"
2. "collaboration interpersonal communication"
3. "40 minute assessment duration"

**Why This Works:** Each facet targets a specific requirement, preventing any single skill from dominating the search.

**Implementation:** src/query_deconstructor_v2.py

- Prompt engineering to extract technical skills, behavioral traits, constraints
- Returns 3 to 7 focused search queries
- Handles rate limits with exponential backoff

---

### Stage 2: Multi-Vector Semantic Search

**Goal:** Find candidates matching EACH facet independently

**How:**

1. Generate query embedding for each facet using Gemini embedding-001
2. Search FAISS index (353 pre-computed assessment embeddings)
3. Retrieve top-K candidates per facet (default: 10)
4. Merge all candidates, removing duplicates

**Example:**

Facet 1 ("Java technical"): Finds 10 Java assessments
Facet 2 ("collaboration"): Finds 10 interpersonal assessments
Facet 3 ("40 minutes"): Finds 10 time-appropriate assessments

Merged Pool: 25 to 40 unique candidates (duplicates removed)

**Why This Works:**

- Ensures diverse candidates covering ALL aspects of query
- Avoids bias toward dominant terms
- Increases recall by expanding search surface

**Technical Details:**

- Uses FAISS for efficient cosine similarity search
- Pre-computed embeddings avoid API quota issues
- 500ms delay between searches to respect rate limits

**Implementation:** src/multi_vector_retriever_v2.py (semantic_search method)

---

### Stage 3: LLM Reranking

**Goal:** Select best 10 assessments ensuring balanced coverage

**How:** Use Gemini 2.5 Flash Lite to rerank 25 to 40 candidates based on:

1. **Relevance** - How well does assessment match original query?
2. **Balance** - Does final list cover ALL requirements (technical plus behavioral)?
3. **Diversity** - Avoid 10 similar tests

**Example:**

Input: 30 candidates (15 Java tests, 10 collaboration tests, 5 mixed)

LLM Prompt: "Select 10 most relevant for Java developers who collaborate. MUST include BOTH technical and behavioral tests."

Output: Ranked indices [5, 12, 3, 18, 7, 1, 14, 9, 22, 16]

Final Results: 6 Java tests plus 3 collaboration tests plus 1 mixed

**Why This Works:**

- LLM understands nuanced requirements better than similarity scores
- Explicit instructions ensure balanced recommendations
- Prevents all-technical or all-behavioral results

**Implementation:** src/llm_reranker_v2.py

- Builds reranking prompt with candidate metadata
- Parses LLM response to extract ranked indices
- Handles rate limits with retry logic

---

### Why Multi-Vector Works Better

| Aspect               | Single-Vector                      | Multi-Vector                                                      |
| -------------------- | ---------------------------------- | ----------------------------------------------------------------- |
| Query Understanding  | Embeds entire query as blob        | Decomposes into focused facets                                    |
| Search Coverage      | One search pass                    | Multiple targeted searches                                        |
| Balance              | Biased toward dominant terms       | Ensures all facets represented                                    |
| Typical Recall at 10 | approximately 25 to 35 percent     | 38 to 68 percent (avg approximately 50 to 55 percent)             |
| Deterministic        | Yes (same results every time)      | No (LLM introduces variance)                                      |
| Example              | "Java developer" → only Java tests | "Java developer who collaborates" → Java plus collaboration tests |

---

## Evaluation Metrics

### Recall at K

**Definition:** Percentage of relevant assessments found in top-K predictions.

**Formula:**

Recall at K = (Number Relevant in Top K) / (Number Total Relevant)

**Example:**

- Query: "Java developer"
- Ground Truth: 5 relevant assessments
- Top-10 Predictions: Contains 3 of the 5
- Recall at 10 = 3/5 = 60 percent

### Why Recall at 10?

- **K equals 10** - UI shows 10 recommendations, matching real-world usage
- **Recall over Precision** - Better to show some irrelevant (user can ignore) than miss relevant assessments
- **Industry Standard** - Common metric for recommendation systems

### Running Evaluation

Use the provided Jupyter notebook for comprehensive evaluation:

```bash
# Start Jupyter
jupyter notebook evaluation_and_predictions.ipynb

# Run all cells to:
# 1. Import libraries and load retriever
# 2. Evaluate on training data (10 queries)
# 3. Calculate mean recall and per-query breakdown
# 4. Generate test predictions (9 queries)
# 5. Export results to CSV
```

**Expected Output (varies due to LLM non-determinism):**

```
================================================================================
EVALUATION RESULTS
================================================================================
Mean Recall at 10: 52.33 percent

Per-Query Results:
--------------------------------------------------------------------------------
Query 1: Java developers who collaborate...
  Recall at 10: 60.00 percent
  Matches: 3/5

Query 2: Sales professional entry-level...
  Recall at 10: 33.33 percent
  Matches: 3/9

Note: Results may vary plus or minus 10 to 15 percent between runs due to LLM randomness
================================================================================
```

**Output Files:**

- evaluation_predictions.csv - Training predictions with ground truth comparison
- test_predictions.csv - Test set predictions (90 rows: 9 queries times 10 recommendations)

### Evaluation Data

Located in: ../Gen_AI Dataset/Train-Set-Table 1.csv and Test-Set-Table 1.csv

**Training Set Structure:**

- 10 recruiter queries
- Each query has 5 to 10 ground truth URLs
- Queries cover diverse roles: technical, behavioral, management

**Test Set Structure:**

- 9 recruiter queries without ground truth
- Used for generating final predictions

---

## API Documentation

### Base URL

http://localhost:8000

### Endpoints

#### 1. Health Check

**GET /health**

**Response (200 OK):**

```json
{
  "status": "ok",
  "message": "Service is running"
}
```

#### 2. Get Recommendations

**POST /recommend**

**Request Body:**

```json
{
  "query": "I need Python developers who are good at problem solving",
  "top_k": 10
}
```

**Parameters:**

- query (string, required): Natural language job description or requirements
- top_k (integer, optional): Number of recommendations (1 to 10, default: 10)

**Response (200 OK):**

```json
{
  "query": "I need Python developers who are good at problem solving",
  "recommendations": [
    {
      "name": "Python (New)",
      "url": "https://www.shl.com/solutions/products/assessments/...",
      "description": "Multi-choice test that measures the knowledge of Python programming...",
      "test_types": ["K"],
      "duration_minutes": 11,
      "job_levels": [
        "Mid-Professional",
        "Professional",
        "Individual Contributor"
      ],
      "score": 0.891
    }
  ],
  "total": 10
}
```

**Error Response (503 Service Unavailable):**

```json
{
  "detail": "Service not ready - retriever not initialized"
}
```

**Error Response (500 Internal Server Error):**

```json
{
  "detail": "Error generating recommendations: error message"
}
```

### Interactive API Docs

Visit http://localhost:8000/docs for Swagger UI with:

- Live API testing
- Request and response schemas
- Example payloads
- Try it out functionality

---

## Frontend

### Architecture

- **Framework:** Next.js 14 with App Router
- **Styling:** Tailwind CSS with SHL brand colors
- **State Management:** React hooks (useState)
- **API Integration:** Direct fetch to backend at http://localhost:8000/recommend

### Design System

**SHL Brand Colors:**

- Primary Green: hex 78D64B (Tailwind: green-600)
- Dark Gray: hex 4a4a4a (Tailwind: gray-900)
- Light Backgrounds: White and light grays
- Accent: Green for buttons, highlights, and interactive elements

### Components

#### app/page.tsx - Main Page

**Features:**

- SHL logo header (SVG from /public/shl-logo.svg)
- Hero section with title and description
- Search interface integration
- Results display area
- Footer with copyright
- Error handling with user-friendly messages
- Loading states during API calls

**State Management:**

```typescript
const [results, setResults] = useState<any[]>([]);
const [loading, setLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
```

#### components/SearchBar.tsx

**Features:**

- Controlled input component
- Search icon (magnifying glass SVG)
- Green submit button with hover effects
- Loading spinner animation during search
- Form validation (non-empty query)
- Disabled states during loading

**Props:**

```typescript
interface SearchBarProps {
  onSearch: (query: string) => void;
  loading: boolean;
}
```

**Styling:**

- Rounded borders with focus rings
- Green color scheme (bg-green-600, hover:bg-green-700)
- Responsive width (max-w-4xl)

#### components/ResultsDisplay.tsx

**Features:**

- Displays top 10 assessment cards
- Shows: name, description, test types, duration, job levels
- Numbered list (1 to 10) with green badges
- Clickable "View Assessment" links (open in new tab)
- Loading skeleton while fetching
- Empty state with icon and message

**Props:**

```typescript
interface ResultsDisplayProps {
  results: Assessment[];
  loading: boolean;
}

interface Assessment {
  name: string;
  url: string;
  description?: string;
  test_types?: string[];
  duration_minutes?: number;
  job_levels?: string[];
  score?: number;
}
```

**Styling:**

- Card-based layout with shadows
- Hover effects for better UX
- Badges for test types (blue), duration (gray), job levels (purple)
- Responsive grid on larger screens

### Running Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### Production Build

```bash
cd frontend
npm run build
npm start
# Runs optimized production build
```

### Environment Configuration

Frontend connects to backend at http://localhost:8000/recommend. To change, edit frontend/app/page.tsx:

```typescript
const response = await fetch("http://your-backend-url:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, top_k: 10 }),
});
```

---

## Usage

### Basic Usage

**1. Start Backend:**

```bash
source .venv/bin/activate
python api.py
```

**2. Start Frontend:**

```bash
cd frontend
npm run dev
```

**3. Search for Assessments:**

- Open http://localhost:3000
- Enter query: "Senior data scientist with Python and ML experience"
- Click Search
- View 10 recommended assessments with metadata

### Advanced Usage

#### Programmatic API Access

```python
import requests

response = requests.post(
    'http://localhost:8000/recommend',
    json={
        'query': 'Java developers who can collaborate',
        'top_k': 10
    }
)

recommendations = response.json()['recommendations']
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['name']} ({rec['duration_minutes']} min)")
    print(f"   URL: {rec['url']}")
    print(f"   Types: {', '.join(rec['test_types'])}")
    print()
```

#### Run Evaluation

```bash
# Open Jupyter notebook
jupyter notebook evaluation_and_predictions.ipynb

# Execute all cells to:
# 1. Initialize Multi-Vector Retriever
# 2. Evaluate on training data (10 queries)
# 3. Calculate Recall at 10 metrics
# 4. Generate test predictions (9 queries)
# 5. Export CSV files

# Note: Expect 10 to 15 percent variance between runs due to LLM non-determinism
```

#### Scraping Updated Catalog

```bash
# Scrape latest SHL catalog
python scrape_shl_catalog.py

# Outputs to:
# - output/shl_catalog.json
# - output/shl_catalog.csv
```

#### Rebuilding Index

```bash
# After updating catalog or embeddings
python build_index.py

# Rebuilds:
# - data/gemini/faiss_gemini.index
# - data/gemini/embeddings_gemini.npy
# - data/gemini/metadata_gemini.json
```

---

## Environment Variables

Required in .env file:

```
GEMINI_API_KEY=your_api_key_here
```

**Getting API Key:**

1. Visit https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click Create API Key
4. Copy and paste into .env

**Free Tier Limits:**

- Embeddings: 100 requests per minute
- Text Generation: 10 requests per minute
- Note: System includes automatic retry logic with exponential backoff

---

## Contributing

This is an educational project. Contributions welcome!

1. Fork repository
2. Create feature branch: git checkout -b feature-name
3. Commit changes: git commit -m 'Add feature'
4. Push to branch: git push origin feature-name
5. Open Pull Request

---

## License

This project is for educational and research purposes. Please respect SHL's website terms of use when scraping data.

---

## Acknowledgments

- **SHL** for comprehensive assessment catalog
- **Google** for Gemini API access
- **Facebook Research** for FAISS library
- **Next.js** and **FastAPI** teams for excellent frameworks
- **Vercel** for Next.js deployment platform

---

**Built with love by Saksham**

**Repository:** github.com/Sakshampoply/shl-product-catalog
