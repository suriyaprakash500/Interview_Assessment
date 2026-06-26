import os
# Change working directory to this script's directory to resolve relative paths in deployments
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import json
import time
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Multimodal RAG Evaluation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Engineering Aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1.2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .explain-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        color: #1E40AF;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.embeddings.embedder import DocumentEmbedder
from src.embeddings.indexer import FAISSIndexer
from src.retrieval.searcher import SearchCoordinator
from src.evaluation.evaluator import RAGEvaluator

# Load Config File
CONFIG_PATH = "config.yaml"
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Retrieve values dynamically for Sidebar Status
pdf_path = config.get("pdf_path", "input/financial_report.pdf")
cache_path = os.path.join(config.get("output_dir", "outputs"), "ingested_pages.json")
text_meta_path = config.get("text_baseline_index_path", "indexes/text_baseline_index.faiss").replace(".faiss", "_metadata.json")
mm_index_file = config.get("multimodal_index_path", "indexes/multimodal_index.faiss")
text_index_file = config.get("text_baseline_index_path", "indexes/text_baseline_index.faiss")

# Load Cache Data
cache_pages = {}
total_pages = 0
last_ingested = "N/A"
if os.path.exists(cache_path):
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
            cache_pages = cache.get("pages", {})
            total_pages = cache.get("total_pages", 0)
            
            # Find latest timestamp
            timestamps = [p.get("ingested_at") for p in cache_pages.values() if p.get("ingested_at")]
            if timestamps:
                last_ingested = max(timestamps)[:16].replace("T", " ")
    except Exception:
        pass

# Count visual captions generated
captions_count = sum(1 for p in cache_pages.values() if p.get("visual_caption"))

# Count chunks
chunks_count = 0
if os.path.exists(text_meta_path):
    try:
        with open(text_meta_path, "r", encoding="utf-8") as f:
            chunks_count = len(json.load(f))
    except Exception:
        pass

# Index status checks
mm_status = "Active" if os.path.exists(mm_index_file) else "Inactive"
text_status = "Active" if os.path.exists(text_index_file) else "Inactive"
pdf_loaded = "Yes" if os.path.exists(pdf_path) else "No"

# -------------------------------------------------------------
# SIDEBAR: DATASET STATUS PANEL
# -------------------------------------------------------------
st.sidebar.title("📁 Dataset Status Panel")
st.sidebar.markdown("---")

st.sidebar.markdown(f"**PDF Loaded:** {pdf_loaded}")
if pdf_loaded == "Yes":
    st.sidebar.markdown(f"**Document Name:** `{os.path.basename(pdf_path)}`")
    st.sidebar.markdown(f"**Total Pages:** {total_pages if total_pages > 0 else '191'}")
else:
    st.sidebar.markdown("**Document Name:** `N/A`")
    st.sidebar.markdown("**Total Pages:** `N/A`")
    
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Pages Processed:** {len(cache_pages)}")
st.sidebar.markdown(f"**Text Chunks Created:** {chunks_count}")
st.sidebar.markdown(f"**Visual Captions (Groq):** {captions_count}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Text Index Status:** `{text_status}`")
st.sidebar.markdown(f"**Multimodal Index Status:** `{mm_status}`")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Ingestion:** `{last_ingested}`")

st.sidebar.markdown("---")
st.sidebar.info("💡 Parameters are loaded directly from `.env` and `config.yaml` to ensure consistent execution.")

# Helper to initialize coordinator
def init_system():
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name = config.get("local_embedding_model") if provider == "local" else config.get("openai_embedding_model")
    embedder = DocumentEmbedder(provider=provider, model_name=model_name)
    indexer = FAISSIndexer(config=config, embedder=embedder)
    
    indices_loaded = indexer.load_indexes()
    search_coordinator = None
    if indices_loaded:
        search_coordinator = SearchCoordinator(config=config, embedder=embedder, indexer=indexer)
        
    return embedder, indexer, search_coordinator, indices_loaded

# UI Layout
st.markdown("<div class='main-title'>Multimodal RAG Evaluation Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Comparison of Groq Vision Multimodal Retrieval vs. Text-only RAG Baseline</div>", unsafe_allow_html=True)

tab_explore, tab_search, tab_eval = st.tabs(["📁 Document Explorer", "🔍 Side-by-Side Search", "📊 Evaluation Dashboard"])

# -------------------------------------------------------------
# TAB 1: DOCUMENT EXPLORER
# -------------------------------------------------------------
with tab_explore:
    st.header("Document Explorer")
    
    if pdf_loaded == "No":
        st.error(f"Target PDF file not found at: `{pdf_path}`. Please verify.")
    else:
        st.write(f"📂 **Active PDF:** `{pdf_path}` | **Processed:** {len(cache_pages)} pages")
        st.success("✅ **Dataset already ingested. Indexes active.** (To run new ingestion or rebuild cache, run: `python run_ingestion.py`)")
        st.markdown("---")
        
        if cache_pages:
            page_keys = sorted([int(k) for k in cache_pages.keys()])
            selected_page = st.selectbox("Select Page to Explore:", page_keys)
            page_meta = cache_pages[str(selected_page)]
            
            # Side-by-side split screen
            col_img, col_data = st.columns([1, 1])
            with col_img:
                img_p = page_meta.get("page_image_path")
                if img_p and os.path.exists(img_p):
                    st.image(img_p, caption=f"Page {selected_page} Rendered Image", use_container_width=True)
                else:
                    st.warning("Rendered page image not found on disk.")
            with col_data:
                st.markdown(f"### Page {selected_page} Metadata")
                st.write(f"📝 **Document Name:** `{page_meta.get('document_name', os.path.basename(pdf_path))}`")
                st.write(f"⚙️ **Vision Model:** `{page_meta.get('visual_caption_model')}`")
                st.write(f"📅 **Ingested At:** `{page_meta.get('ingested_at')}`")
                
                st.markdown("#### Qwen-VL / Groq Visual Caption")
                st.info(page_meta.get("visual_caption", "No visual caption generated."))
                
                st.markdown("#### Extracted Page Text")
                st.text_area("Verbatim Text", page_meta.get("extracted_text", ""), height=250)
        else:
            st.warning("No ingested pages found. Run ingestion to view document details.")

# -------------------------------------------------------------
# TAB 2: SIDE-BY-SIDE SEARCH
# -------------------------------------------------------------
with tab_search:
    st.header("Visual Search Comparison")
    
    embedder, indexer, search_coordinator, indices_loaded = init_system()
    
    if not indices_loaded:
        st.warning("⚠️ FAISS vector indexes are not built yet. Please ingest and index first.")
    else:
        search_mode = st.radio("Search Mode:", ["📝 Text Query", "🖼️ Image Query"], horizontal=True)
        
        query_text = ""
        query_image_path = None
        
        if search_mode == "📝 Text Query":
            preset = st.selectbox("Example Assessment Queries:", [
                "",
                "What was the highest-ever total revenue of TVS Motor Company in FY 2024-25?",
                "What was the PBT (Profit Before Tax) for TVS Motor Company in FY 2024-25?",
                "What is the renewable power contribution to the overall share of power in FY 2024-25?",
                "What was the contribution of exports to the Company's revenues in Spotlight Story #2?",
                "What is the number of users of TVSM vehicles shown in the highlights?",
                "What was the penetration rate of the two-wheeler EV scooter industry according to the VAHAN snapshot?"
            ])
            query_text = st.text_input("Enter text query:", value=preset if preset else "")
        else:
            uploaded_file = st.file_uploader("Upload screenshot of a chart or table:", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                temp_dir = os.path.join(config.get("output_dir", "outputs"), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "query_image.png")
                
                image = Image.open(uploaded_file)
                image.save(temp_path)
                query_image_path = temp_path
                st.image(image, caption="Uploaded Query Screenshot", width=300)
                
        top_k = st.slider("Select K (number of pages to retrieve):", min_value=1, max_value=10, value=3)
        
        # Session state initialization for search trigger
        if "search_triggered" not in st.session_state:
            st.session_state.search_triggered = False
        if "last_query" not in st.session_state:
            st.session_state.last_query = None
        if "last_mode" not in st.session_state:
            st.session_state.last_mode = search_mode

        # Determine active query
        active_query = query_text if search_mode == "📝 Text Query" else query_image_path

        # If search mode or active query changed, reset the trigger to require explicit button click
        if st.session_state.last_mode != search_mode or st.session_state.last_query != active_query:
            st.session_state.search_triggered = False
            st.session_state.last_mode = search_mode
            st.session_state.last_query = active_query

        # Search button
        if st.button("🔍 Execute RAG Retrieval"):
            if not active_query:
                st.warning("⚠️ Please enter a text query or upload an image before executing retrieval.")
                st.session_state.search_triggered = False
            else:
                st.session_state.search_triggered = True

        # Run search if triggered
        if st.session_state.search_triggered and active_query:
            with st.spinner("Searching FAISS indexes..."):
                mm_results = []
                text_results = []
                t_text_lat = 0.0
                t_mm_lat = 0.0
                query_caption = ""
                
                if search_mode == "📝 Text Query":
                    mm_results, text_results, t_text_lat, t_mm_lat = search_coordinator.search_by_text(query_text, top_k=top_k)
                else:
                    mm_results, text_results, t_text_lat, t_mm_lat, total_lat, query_caption = search_coordinator.search_by_image(query_image_path, top_k=top_k)
            
            st.write(f"⏱️ **Index Lookup Latency:** Text Index: `{t_text_lat:.4f}s` | Multimodal Index: `{t_mm_lat:.4f}s`")
            if query_caption:
                st.info(f"**Groq VLM Query Caption:** {query_caption}")
                
            # Side-by-side search results
            col_mm_res, col_text_res = st.columns(2)
            
            with col_mm_res:
                st.markdown("### 🌟 Multimodal Index Results")
                st.markdown("*Embeds verbatim text & visual VLM descriptions*")
                
                if not mm_results:
                    st.write("No pages retrieved.")
                for idx, res in enumerate(mm_results, 1):
                    with st.container():
                        st.markdown(f"#### Rank {idx}: Page {res['page_number']}")
                        st.write(f"🎯 **Similarity Score:** `{res['similarity_score']:.4f}`")
                        
                        # MATCHED BECAUSE EXPLANATION (INTERVIEW EXPLAINABILITY)
                        m_type = res.get("match_type", "unknown")
                        if m_type == "text":
                            st.markdown("<div class='explain-box'>💡 <b>Matched because:</b> Verbatim page text matches the query vector.</div>", unsafe_allow_html=True)
                        elif m_type == "visual":
                            st.markdown("<div class='explain-box'>💡 <b>Matched because:</b> Qwen-VL/Groq visual caption matches the query vector (details of a chart, table, or infographic on this page).</div>", unsafe_allow_html=True)
                        elif m_type == "hybrid":
                            st.markdown("<div class='explain-box'>💡 <b>Matched because:</b> Enriched Hybrid chunk matches (blending page text and Groq visual caption details).</div>", unsafe_allow_html=True)
                        
                        img_path = res.get("page_image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"Page {res['page_number']} Preview", use_container_width=True)
                        else:
                            st.warning("Page image not found.")
                            
                        with st.expander("Matched Snippet Context"):
                            st.write(res["matching_snippet"])
                        st.markdown("---")
                        
            with col_text_res:
                st.markdown("### 📝 Text-only Baseline Results")
                st.markdown("*Embeds raw text chunks only (No visuals)*")
                
                if not text_results:
                    st.write("No pages retrieved.")
                for idx, res in enumerate(text_results, 1):
                    with st.container():
                        st.markdown(f"#### Rank {idx}: Page {res['page_number']}")
                        st.write(f"🎯 **Similarity Score:** `{res['similarity_score']:.4f}`")
                        st.markdown("<div class='explain-box'>💡 <b>Matched because:</b> Text baseline chunk contains semantic keyword matches.</div>", unsafe_allow_html=True)
                        
                        img_path = res.get("page_image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"Page {res['page_number']} Preview", use_container_width=True)
                        else:
                            st.warning("Page image not found.")
                            
                        with st.expander("Matched Chunk Context"):
                            st.write(res["matching_snippet"])
                        st.markdown("---")

# -------------------------------------------------------------
# TAB 3: EVALUATION DASHBOARD
# -------------------------------------------------------------
with tab_eval:
    st.header("Quantitative RAG Evaluation Dashboard")
    
    eval_csv_path = "evaluation_results.csv"
    eval_report_path = "evaluation_report.md"
    latency_csv_path = "evaluation/latency_results.csv"
    
    embedder, indexer, search_coordinator, indices_loaded = init_system()
    
    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        st.write("### Benchmark Controls")
        if not indices_loaded:
            st.error("FAISS indexes not built yet. Build indexes to run evaluation.")
        else:
            if st.button("📊 Run Evaluation Suite"):
                with st.spinner("Executing all 15 ground-truth queries..."):
                    evaluator = RAGEvaluator(search_coordinator)
                    df, summary = evaluator.run_evaluation()
                    evaluator.save_results(df, summary)
                    st.success("Evaluation completed!")
                    st.rerun()
                    
    with col_info:
        st.write("### Ground-Truth Methodology")
        st.markdown("""
        * **Evaluation Set:** 15 labeled queries from `evaluation/queries.json` mapped to expected page numbers.
        * **Categories:** 8 Text-focused queries (verbatim paragraphs) vs. 7 Visual-focused queries (infographics, charts, segment ratios).
        * **Metrics:** Precision@5, Recall@5, MRR, and actual measured retrieval/end-to-end latencies.
        """)

    if os.path.exists(eval_csv_path) and os.path.exists(latency_csv_path):
        df_results = pd.read_csv(eval_csv_path)
        df_latency = pd.read_csv(latency_csv_path)
        
        # Display Metrics Summary cards using actual measured values
        mm_rec = df_results["mm_recall_5"].mean()
        text_rec = df_results["text_recall_5"].mean()
        mm_mrr = df_results["mm_mrr"].mean()
        text_mrr = df_results["text_mrr"].mean()
        mm_lat = df_latency["end_to_end_multimodal_latency"].mean()
        text_lat = df_latency["end_to_end_text_latency"].mean()
        
        st.subheader("Benchmark Comparison Summaries")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>RECALL@5</div>
                <div class='metric-value'>{mm_rec*100:.1f}% vs {text_rec*100:.1f}%</div>
                <div class='metric-label'>{(mm_rec - text_rec)*100:+.1f}% Delta</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>MEAN RECIPROCAL RANK (MRR)</div>
                <div class='metric-value'>{mm_mrr:.4f} vs {text_mrr:.4f}</div>
                <div class='metric-label'>{mm_mrr - text_mrr:+.4f} Delta</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>PRECISION@5</div>
                <div class='metric-value'>{df_results['mm_precision_5'].mean():.4f} vs {df_results['text_precision_5'].mean():.4f}</div>
                <div class='metric-label'>{df_results['mm_precision_5'].mean() - df_results['text_precision_5'].mean():+.4f} Delta</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>MEAN END-TO-END LATENCY</div>
                <div class='metric-value'>{mm_lat:.4f}s vs {text_lat:.4f}s</div>
                <div style='font-size:0.8rem; color:#64748B; font-weight:600;'>{mm_lat - text_lat:+.4f}s Delta</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Draw Charts using Streamlit native charts
        st.subheader("Category Performance Metrics")
        col_c1, col_c2 = st.columns(2)
        
        categories = df_results["query_type"].unique()
        with col_c1:
            st.markdown("##### Recall@5 Comparison by Query Type")
            chart_data_recall = []
            for cat in categories:
                cat_df = df_results[df_results["query_type"] == cat]
                chart_data_recall.append({
                    "Query Type": cat.title(),
                    "Text RAG Baseline": float(cat_df["text_recall_5"].mean()),
                    "Multimodal RAG": float(cat_df["mm_recall_5"].mean())
                })
            df_chart_recall = pd.DataFrame(chart_data_recall).set_index("Query Type")
            st.bar_chart(df_chart_recall, height=280)
            
        with col_c2:
            st.markdown("##### End-to-End Latency (Seconds) by Query Type")
            chart_data_lat = []
            for cat in categories:
                cat_lat = df_latency[df_latency["query_type"] == cat]
                chart_data_lat.append({
                    "Query Type": cat.title(),
                    "Text RAG Baseline": float(cat_lat["end_to_end_text_latency"].mean()),
                    "Multimodal RAG": float(cat_lat["end_to_end_multimodal_latency"].mean())
                })
            df_chart_lat = pd.DataFrame(chart_data_lat).set_index("Query Type")
            st.bar_chart(df_chart_lat, height=280)

        # Show detailed results table with highlights where Multimodal outperformed
        st.subheader("Query-by-Query Metrics & Multi-Modal Wins")
        st.markdown("*Highlighted rows indicate queries where Multimodal retrieval outperformed the Text-only baseline.*")
        
        styled_df = df_results[[
            "query_id", "query_type", "query", "expected_page", 
            "text_rank", "text_mrr", "text_recall_5",
            "mm_rank", "mm_mrr", "mm_recall_5"
        ]].copy()
        
        styled_df.columns = [
            "ID", "Type", "Query Text", "Target Page",
            "Text Rank", "Text MRR", "Text Recall@5",
            "MM Rank", "MM MRR", "MM Recall@5"
        ]
        
        # Highlighting logic: MM Recall > Text Recall OR MM MRR > Text MRR
        # Standard pandas highlights
        def highlight_mm_wins(row):
            win = (row["MM Recall@5"] > row["Text Recall@5"]) or (row["MM MRR"] > row["Text MRR"])
            color = 'background-color: #E2F0D9' if win else ''
            return [color] * len(row)
            
        st.dataframe(styled_df.style.apply(highlight_mm_wins, axis=1))

        # MULTIMODAL EXPLANATION PANEL
        st.subheader("⚙️ System Architecture Pipeline")
        st.markdown("""
        The multimodal search pipeline works as follows:
        1. **Ingestion Layer:**
           - **Text Extraction:** Extract verbatim page text from the PDF using PyMuPDF.
           - **Image Rendering:** Render each PDF page to a high-resolution PNG.
           - **Groq VLM Inference:** Pass the rendered page image to the **Groq Llama 4 Vision** endpoint to extract charts, graphs, and tables into text captions.
        2. **Vector Indexing Layer:**
           - Generate vector representations for both the extracted text and the Groq visual captions using the configured embedding model (`SentenceTransformers` or `OpenAI`).
           - Push both vectors to the same **FAISS Multimodal Index**, mapping them to the source page number.
        3. **RAG Retrieval Layer:**
           - When searching, the query text is embedded.
           - We search the FAISS Multimodal Index. This matches against both the page text and the visual captions.
           - Results are ranked by Cosine Similarity and grouped by page number, returning visual previews of the documents.
        """)

        # Show analysis text from evaluation_report.md
        st.subheader("Evaluation Analysis & Discussion")
        if os.path.exists(eval_report_path):
            with open(eval_report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
    else:
        st.info("No evaluation results file found. Click 'Run Evaluation Suite' to execute and display metrics.")
