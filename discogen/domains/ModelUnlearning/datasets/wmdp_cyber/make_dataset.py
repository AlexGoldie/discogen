import zipfile
import os
import urllib.request
import subprocess

def make_dataset():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    # Check if extracted dataset folder already exists
    extracted_folder = os.path.join(dest_dir, "wmdp-corpora")
    if os.path.exists(extracted_folder):
        print(f"Dataset already exists in {extracted_folder}, skipping download.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    # Download zip
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded to {zip_path}")

    # Try fast system unzip first, otherwise Python unzip
    print("Extracting password-protected zip...")
    try:
        subprocess.run(["unzip", "-P", "wmdpcorpora", "-o", zip_path, "-d", dest_dir], check=True)
        print(f"Dataset created in {dest_dir}.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("System unzip not available, using Python's zipfile...")
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.setpassword(b'wmdpcorpora')
            zip_file.extractall(dest_dir)
        print(f"Dataset created in {dest_dir}")
