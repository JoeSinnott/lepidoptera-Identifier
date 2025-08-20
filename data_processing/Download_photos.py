import pandas as pd
import dask.dataframe as dd
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Configuration ---
BASE_OUTPUT_DIR = 'dataset'
PARQUET_FOLDER = 'photo_id_data.parquet' # The folder created by your first script

# --- Worker Function (for multiprocessing) ---
# This code will be run by each parallel worker process.
s3_client = None

def initialize_worker():
    """Initializes a persistent S3 client for each worker process."""
    global s3_client
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_image(args):
    """Downloads a single image using the worker's S3 client."""
    photo_id, species_name, extension, base_dir = args
    
    if not (pd.notna(photo_id) and pd.notna(species_name)):
        return
    if not isinstance(extension, str) or extension.lower() not in ['jpg', 'jpeg']:
        extension = 'jpg'
        
    folder_name = str(species_name).replace(' ', '_')
    output_folder = os.path.join(base_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    local_path = os.path.join(output_folder, f"{int(photo_id)}.jpg")
    
    if os.path.exists(local_path):
        return
        
    bucket_name = 'inaturalist-open-data'
    s3_key = f"photos/{int(photo_id)}/medium.{extension}"
    
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
    except ClientError:
        # Silently skip if the file is not found or another error occurs.
        pass

# --- Main Execution Logic ---
if __name__ == '__main__':
    # --- Step 1: Load the clean, pre-processed data from Parquet files ---
    # This is extremely fast and memory-efficient.
    print(f"Loading pre-processed data from '{PARQUET_FOLDER}'...")
    try:
        df_to_process = dd.read_parquet(PARQUET_FOLDER, engine='pyarrow')
    except FileNotFoundError:
        print(f"ERROR: The directory '{PARQUET_FOLDER}' was not found.")
        print("Please run the first data preparation script to create it.")
        exit()

    # --- Step 2: The "Dask to Pandas Bridge" for downloading ---
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    num_workers = cpu_count()
    print(f"Starting download process with {num_workers} parallel workers...")

    # Create the pool of workers ONCE.
    with Pool(processes=num_workers, initializer=initialize_worker) as pool:
        
        # Iterate through the Dask DataFrame one partition at a time.
        for i, partition in enumerate(df_to_process.to_delayed()):
            print(f"\n--- Processing Partition {i+1} ---")
            
            # Compute ONLY the current partition to get a REAL pandas DataFrame.
            # This will be very fast because it's reading from efficient Parquet files.
            pdf_partition = partition.compute()
            
            # Create a task generator from the simple pandas DataFrame.
            tasks_generator = (
                (row.photo_id, row.name, row.extension, BASE_OUTPUT_DIR) 
                for row in pdf_partition.itertuples(index=False)
            )
            
            # Process the tasks for this partition in parallel.
            list(tqdm(pool.imap_unordered(download_image, tasks_generator), total=len(pdf_partition)))

    print("\nDownload process complete.")
