import os
import yaml
import logging
from dotenv import load_dotenv
from src.embeddings.embedder import DocumentEmbedder
from src.embeddings.indexer import FAISSIndexer
from src.retrieval.searcher import SearchCoordinator
from src.evaluation.evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_evaluation")

def main():
    # Load env vars
    load_dotenv()
    
    # Load config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Initialize Embedder
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name = config.get("local_embedding_model") if provider == "local" else config.get("openai_embedding_model")
    embedder = DocumentEmbedder(provider=provider, model_name=model_name)
    
    # Initialize Indexer and Load Indexes
    indexer = FAISSIndexer(config=config, embedder=embedder)
    if not indexer.load_indexes():
        logger.error("Could not load FAISS indexes. Please run ingestion and indexing first!")
        return
        
    # Initialize Search Coordinator
    search_coordinator = SearchCoordinator(config=config, embedder=embedder, indexer=indexer)
    
    # Initialize Evaluator
    evaluator = RAGEvaluator(search_coordinator=search_coordinator)
    
    try:
        # Run evaluation
        df, summary = evaluator.run_evaluation()
        
        # Save results
        evaluator.save_results(df, summary)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS OVERALL SUMMARY:")
        print("="*50)
        overall = summary["overall"]
        print(f"{'Metric':<20} | {'Text RAG':<12} | {'Multimodal RAG':<14} | {'Improvement':<12}")
        print("-"*65)
        for m in ["precision_5", "recall_5", "mrr", "latency"]:
            name = "Precision@5" if m == "precision_5" else ("Recall@5" if m == "recall_5" else ("MRR" if m == "mrr" else "Latency (s)"))
            t_val = overall["text"][m]
            m_val = overall["mm"][m]
            diff = f"{m_val - t_val:+.4f}s" if m == "latency" else f"{(m_val - t_val)*100:+.1f}%"
            print(f"{name:<20} | {t_val:<12.4f} | {m_val:<14.4f} | {diff:<12}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
