# Import the argparse module
import argparse
import requests
from huggingface_hub import snapshot_download
# Import the tqdm module
from tqdm import tqdm

# Create the parser
parser = argparse.ArgumentParser(description="A simple script that takes two input arguments and prints them")

# Add the arguments
parser.add_argument("--repo_id", type=str, help="HuggingFace repo id", required=True)
parser.add_argument("--revision", type=str, help="Model revision", default="main")

# Parse the arguments
args = parser.parse_args()

# Print the arguments only when the script is run directly
if __name__=="__main__":
    # Get the number of files to be downloaded
    response = requests.head(f"https://huggingface.co/{args.repo_id}/resolve/{args.revision}/")
    file_count = int(response.headers.get("Content-Length", 0))
    
    # Wrap the snapshot_download function with tqdm function
    for _ in tqdm(snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=f"../models/{args.repo_id}/{args.revision}"), total=file_count):
        pass
