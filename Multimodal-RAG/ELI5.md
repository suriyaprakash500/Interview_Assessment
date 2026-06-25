# ELI5: How Our Multimodal RAG System Works
*(Explain Like I'm 5 — Interview Preparation Guide)*

This guide will help you understand the architecture, trade-offs, and final benchmarks of your Multimodal Retrieval system so you can confidently defend them in a technical interview walkthrough.

---

## 1. The Core Problem: Why Traditional RAG Fails
Imagine you are reading a company's Annual Report. 
* **Traditional RAG (Text-only)** is like reading the document in the dark with a tiny flashlight: it only sees raw text. If there is a beautiful chart showing EV sales growth, or a segment revenue pie chart, the text extractor sees it as a blank space, random graphics, or scrambled characters.
* If a user asks: *"What was the EV scooter industry growth rate in FY25?"*, traditional RAG misses it completely because that data only lives inside a visual chart, not in the text.

---

## 2. Our Solution: Multimodal Retrieval
We solve this by giving the system "eyes" (a hosted Vision-Language Model on Groq):
1. **Text Extraction**: We extract verbatim page text (using PyMuPDF).
2. **Page Rendering**: We render each page as a PNG image.
3. **Groq VLM Inference**: We feed the page image to the vision model (`meta-llama/llama-4-scout-17b-16e-instruct`). It acts like a human reviewer: it looks at the page, reads the charts, extracts the coordinate values, deciphers table grids, and writes a detailed text description (Visual Caption) of everything on that page.
4. **Vector Indexing**: We embed these captions and raw text into a FAISS vector database.

---

## 3. The Two Architectures We Benchmarked
During the evolution of this project, we compared two different indexing strategies:

### Approach A: Page-Level Split Vectors (Original Setup)
* **How it worked**: For each page, we generated **two separate vectors** in the FAISS index: one for the entire page's raw text, and one for the page's visual description.
* **Results**: **Very High Recall (+60%) and MRR (0.3800)**.
* **Why**: The visual caption vector acted as a dedicated "visual portal" to that page. If a query asked about a chart, it matched the visual vector directly, without text interference.

### Approach B: Semantically Enriched Hybrid Chunking (Option 3 - Current Setup)
* **How it works**: We split the page text into small chunks (800 characters). Then, we **merge** the VLM visual caption directly into *every single text chunk* of that page:
  `[Verbatim Text Chunk] + [Visual Image Context: VLM Caption]`
* **Results**: **Lower Recall (46.67%) and MRR (0.2189)**.
* **Why (The "Semantic Dilution" Trade-Off)**: This is the core engineering insight you can present in your interview:
  1. **Signal Dilution**: Appending a long visual caption (~1,000+ characters of table numbers and visual details) to a small text chunk (~800 characters) dilutes the semantic signal. When a user asks a text-specific question, the embedding vector's direction is pulled away by the noisy visual context.
  2. **Vector Space Crowding (Noise)**: Because the same visual caption is appended to *every* chunk on that page, all vectors from that page look nearly identical to the embedder. Retrieval returns multiple redundant chunks from the same page, blocking other relevant pages from entering the Top 5 list.

---

## 4. Key Defense Points for Your Interview

When the interviewer asks you to defend your decisions, use these key points:

### Q1: Why did you use a hosted Vision API (Groq) instead of running a model locally?
* **Defense**: *"Due to hardware constraints (local GTX 1650 with 4GB VRAM), running a large state-of-the-art vision-language model like Qwen2.5-VL-72B locally was not practical. I chose a hosted API architecture on Groq. This allowed us to benchmark visual understanding using a production-grade 17B Llama 4 Vision model with zero local hardware requirements, ensuring consistent, high-fidelity image parsing."*

### Q2: Why did you limit ingestion to 94 pages?
* **Defense**: *"The Groq free-tier key has a strict rate limit of 500,000 Tokens Per Day (TPD). Ingesting high-resolution page images as base64 consumes ~1,500–2,500 input/output tokens per page. We hit this daily limit at page 95. However, the system is designed with an incremental page-level cache (`ingested_pages.json`), meaning that once a paid key or local endpoint is configured, ingestion resumes automatically from page 95 without duplicate costs."*

### Q3: What did you learn from the Hybrid Chunking (Option 3) experiment?
* **Defense**: *"Merging text chunks and page-level visual captions into a single hybrid vector (Option 3) degraded retrieval performance compared to the text-only baseline (Recall fell from 53.3% to 46.7%). This taught us that visual captions dilute the semantic focus of text chunks when combined. In production, a separate multi-index approach (searching text vectors and visual vectors independently and merging results via Reciprocal Rank Fusion) is superior to simple string concatenation."*
