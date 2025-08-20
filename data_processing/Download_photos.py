import pandas as pd
import dask.dataframe as dd
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Where to save the final dataset
BASE_OUTPUT_DIR = 'dataset'
# Folder with the processed photo metadata
PARQUET_FOLDER = 'photo_id_data.parquet' 

# This needs to be global for the worker processes
s3_client = None

def initialize_worker():
    """Create a single S3 client for each worker process to reuse."""
    global s3_client
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_image(args):
    """Downloads an image from S3 if it doesn't already exist."""
    photo_id, species_name, extension, base_dir = args
    
    # Skip rows with missing data
    if not (pd.notna(photo_id) and pd.notna(species_name)):
        return
    # Default to jpg if the extension isnt valid
    if not isinstance(extension, str) or extension.lower() not in ['jpg', 'jpeg']:
        extension = 'jpg'
        
    # Create a folder for the species, replacing spaces with underscores
    folder_name = str(species_name).replace(' ', '_')
    output_folder = os.path.join(base_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    local_path = os.path.join(output_folder, f"{int(photo_id)}.jpg")
    
    # Dont redownload images we already have
    if os.path.exists(local_path):
        return
        
    bucket_name = 'inaturalist-open-data'
    s3_key = f"photos/{int(photo_id)}/medium.{extension}"
    
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
    except ClientError:
        # Skip if the photo doesn't exist on S3 or if there's an error.
        pass

if __name__ == '__main__':
    # Load the metadata from the parquet file.
    print(f"Loading pre-processed data from '{PARQUET_FOLDER}'...")
    try:
        df_to_process = dd.read_parquet(PARQUET_FOLDER, engine='pyarrow')
    except FileNotFoundError:
        print(f"ERROR: The directory '{PARQUET_FOLDER}' was not found.")
        print("Please run the first data preparation script to create it.")
        exit()

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    num_workers = cpu_count()
    print(f"Starting download process with {num_workers} parallel workers...")

    # Set up a pool of workers to download in parallel.
    with Pool(processes=num_workers, initializer=initialize_worker) as pool:
        
        # Process the data one chunk (partition) at a time.
        for i, partition in enumerate(df_to_process.to_delayed()):
            print(f"\n--- Processing Partition {i+1} ---")
            
            # Load the current partition into memory as a pandas DataFrame.
            pdf_partition = partition.compute()
            
            # Prepare the download tasks for this partition.
            tasks_generator = (
                (row.photo_id, row.name, row.extension, BASE_OUTPUT_DIR) 
                for row in pdf_partition.itertuples(index=False)
            )
            
            # Run the downloads for this partition, showing a progress bar.
            list(tqdm(pool.imap_unordered(download_image, tasks_generator), total=len(pdf_partition)))

    print("\nDownload process complete.")