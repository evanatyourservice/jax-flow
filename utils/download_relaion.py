import os
import argparse
from typing import List

import google.cloud.storage
from huggingface_hub import hf_hub_download, list_repo_files

def get_file_list(repo_id: str, repo_type: str = "dataset") -> List[str]:
    """Return the list of files in a Hugging Face dataset repository."""
    return list_repo_files(repo_id=repo_id, repo_type=repo_type)

def upload_file_to_gcs(file_name: str, gcs_bucket: str, gcs_prefix: str) -> bool:
    """
    Download a Parquet file from Hugging Face, upload it to GCS, and remove it locally.
    """
    try:
        print(f"[upload_file_to_gcs] Downloading {file_name}...")
        local_file = hf_hub_download(
            repo_id="laion/relaion2B-en-research-safe",
            filename=file_name,
            repo_type="dataset"
        )
        gcs_filename = os.path.basename(file_name)
        gcs_path = f"{gcs_prefix}/{gcs_filename}"

        print(f"[upload_file_to_gcs] Uploading to gs://{gcs_bucket}/{gcs_path}...")
        storage_client = google.cloud.storage.Client()
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_file)

        if os.path.exists(local_file):
            os.remove(local_file)
        print(f"[upload_file_to_gcs] Finished {file_name}")
        return True
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return False

def upload_all_files_to_gcs(gcs_bucket: str, gcs_prefix: str):
    """Fetch all .snappy.parquet files from Hugging Face and upload them to GCS."""
    print("[upload_all_files_to_gcs] Fetching file list...")
    parquet_files = [
        f for f in get_file_list("laion/relaion2B-en-research-safe")
        if f.endswith(".snappy.parquet")
    ]
    print(f"[upload_all_files_to_gcs] Found {len(parquet_files)} .snappy.parquet file(s)")
    for file_name in parquet_files:
        if upload_file_to_gcs(file_name, gcs_bucket, gcs_prefix):
            print(f"[upload_all_files_to_gcs] Uploaded {file_name}")
        else:
            print(f"[upload_all_files_to_gcs] Failed {file_name}")

def test_first_file(gcs_bucket: str, gcs_prefix: str):
    """
    Test mode: download/upload the first .snappy.parquet file, then check GCS.
    """
    print("[test_first_file] Fetching file list...")
    parquet_files = [
        f for f in get_file_list("laion/relaion2B-en-research-safe")
        if f.endswith(".snappy.parquet")
    ]
    print(f"[test_first_file] Found {len(parquet_files)} .snappy.parquet file(s).")
    if not parquet_files:
        print("[test_first_file] No .snappy.parquet files found.")
        return
    first_file = parquet_files[0]
    print(f"[test_first_file] Using {first_file}...")
    if not upload_file_to_gcs(first_file, gcs_bucket, gcs_prefix):
        print("[test_first_file] Upload failed.")
        return

    gcs_filename = os.path.basename(first_file)
    gcs_path = f"{gcs_prefix}/{gcs_filename}"
    storage_client = google.cloud.storage.Client()
    blob = storage_client.bucket(gcs_bucket).blob(gcs_path)
    if blob.exists():
        print("[test_first_file] File found in GCS.")
    else:
        print("[test_first_file] File not found in GCS.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Relaion dataset .snappy.parquet files to GCS")
    parser.add_argument(
        "--gcs_dir",
        type=str,
        default="gs://diffdata/relaion",
        help="GCS directory (e.g. gs://bucket/prefix)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Upload only the first file, then verify it in GCS."
    )
    args = parser.parse_args()

    if not args.gcs_dir.startswith("gs://"):
        raise ValueError("GCS directory must start with gs://")

    bucket_name = args.gcs_dir.replace("gs://", "").split("/")[0]
    prefix = "/".join(args.gcs_dir.replace("gs://", "").split("/")[1:])

    print(f"Will upload to {args.gcs_dir}/")
    if args.test:
        print("[main] Test mode (single file)...")
        test_first_file(bucket_name, prefix)
    else:
        print("[main] Uploading all .snappy.parquet files...")
        upload_all_files_to_gcs(bucket_name, prefix)
