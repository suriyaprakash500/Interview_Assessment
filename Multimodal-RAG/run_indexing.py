import os
# Change working directory to this script's directory to resolve relative paths in deployments
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import json
import logging
from dotenv import load_dotenv
from src.embeddings.embedder import DocumentEmbedder
from src.embeddings.indexer import FAISSIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_indexing")

def main():
    # Load env vars
    load_dotenv()
    
    # Paths
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    ingested_data_path = os.path.join(config.get("output_dir", "outputs"), "ingested_pages.json")
    if not os.path.exists(ingested_data_path):
        logger.error(f"Ingested pages metadata not found at: {ingested_data_path}. Please run ingestion first!")
        return
        
    with open(ingested_data_path, "r", encoding="utf-8") as f:
        ingested_data = json.load(f)
        
    # Initialize Embedder
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name = config.get("local_embedding_model") if provider == "local" else config.get("openai_embedding_model")
    
    embedder = DocumentEmbedder(provider=provider, model_name=model_name)
    
    # Initialize Indexer
    indexer = FAISSIndexer(config=config, embedder=embedder)
    
    try:
        indexer.build_indexes(ingested_data)
        indexer.save_indexes()
        logger.info("Indexing completed successfully. FAISS indexes saved.")
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)

if __name__ == "__main__":
    main()
