import logging
from config import train_dataset_size, dataset_block_size, filter_with_snr, seed
from src.pipeline import generate_and_save_dataset_in_blocks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetGenerator")

if __name__ == "__main__":
    logger.info("Starting training dataset generation...")
    generate_and_save_dataset_in_blocks(train_dataset_size, dataset_block_size, filter_with_snr=filter_with_snr, seed=seed)
