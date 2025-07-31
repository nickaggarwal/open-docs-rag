#!/usr/bin/env python3
"""
Cleanup script to safely remove data directories

This script removes:
1. Vector store indices
2. Database files
3. Test data
"""

import os
import shutil
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_data(data_path: str = "./data", keep_dirs: bool = False):
    """
    Clean up data directories
    
    Args:
        data_path: Path to data directory
        keep_dirs: Whether to keep directory structure (empty directories)
    """
    logger.info(f"Cleaning up data at {data_path}")
    
    if not os.path.exists(data_path):
        logger.info(f"Data path {data_path} does not exist, nothing to clean")
        return
    
    # Directories to clean
    dirs_to_clean = [
        "faiss_index",
        "real_world_test_db",
        "chroma_db",
        "cache"
    ]
    
    try:
        # Find all relevant directories
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(data_path, dir_name)
            if os.path.exists(dir_path):
                if keep_dirs:
                    # Remove contents but keep directory
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            logger.info(f"Removed file: {item_path}")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            logger.info(f"Removed directory: {item_path}")
                else:
                    # Remove entire directory
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed directory: {dir_path}")
                    # Recreate empty directory
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Recreated empty directory: {dir_path}")
        
        # Remove SQLite database files
        for file in os.listdir(data_path):
            if file.endswith('.db') or file.endswith('.sqlite'):
                file_path = os.path.join(data_path, file)
                os.remove(file_path)
                logger.info(f"Removed database file: {file_path}")
                
        logger.info("Data cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def deep_clean(data_path: str = "./data"):
    """
    Perform a deep clean by removing all data directories completely
    
    Args:
        data_path: Path to data directory
    """
    logger.info(f"Performing deep clean of {data_path}")
    
    if not os.path.exists(data_path):
        logger.info(f"Data path {data_path} does not exist, nothing to clean")
        return
        
    try:
        # Remove the entire data directory
        shutil.rmtree(data_path)
        logger.info(f"Removed data directory: {data_path}")
        
        # Recreate the data directory
        os.makedirs(data_path, exist_ok=True)
        logger.info(f"Recreated empty data directory: {data_path}")
        
        # Recreate standard subdirectories
        standard_dirs = [
            "faiss_index",
            "real_world_test_db",
            "chroma_db"
        ]
        
        for dir_name in standard_dirs:
            dir_path = os.path.join(data_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Recreated empty directory: {dir_path}")
            
        logger.info("Deep clean complete")
        
    except Exception as e:
        logger.error(f"Error during deep clean: {str(e)}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up data directories")
    parser.add_argument("--data-path", default="./data", help="Path to data directory")
    parser.add_argument("--keep-dirs", action="store_true", help="Keep directory structure")
    parser.add_argument("--deep-clean", action="store_true", help="Perform a deep clean (removes everything)")
    args = parser.parse_args()
    
    if args.deep_clean:
        deep_clean(args.data_path)
    else:
        cleanup_data(args.data_path, args.keep_dirs) 