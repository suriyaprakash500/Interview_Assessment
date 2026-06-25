import os
import argparse
import yaml
import logging
from dotenv import load_dotenv
from src.ingestion.pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_ingestion")

def main():
    # Load env vars
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the PDF Ingestion Pipeline.")
    parser.add_argument("--start", type=int, help="Start page (1-indexed)")
    parser.add_argument("--end", type=int, help="End page (1-indexed)")
    parser.add_argument("--limit", type=int, help="Limit number of pages to ingest (from start)")
    parser.add_argument("--mock", action="store_true", help="Force running Qwen-VL in mock mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
        
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Resolve range if limit is set
    start_page = args.start
    end_page = args.end
    
    if args.limit:
        start_page = start_page or 1
        end_page = start_page + args.limit - 1
        logger.info(f"Limit option set. Ingesting {args.limit} pages starting from page {start_page}.")
        
    pipeline = IngestionPipeline(config=config, mock_mode=args.mock)
    
    try:
        pipeline.run(start_page=start_page, end_page=end_page)
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Error running ingestion pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()
