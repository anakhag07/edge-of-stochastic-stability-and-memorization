import os
from pathlib import Path
import urllib.request
import scipy.io as sio
from torchvision import datasets
import torch
import json

if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")

DATASET_FOLDER = Path(os.environ.get('DATASETS'))


def download_cifar10():
    """Download CIFAR-10 dataset using torchvision"""
    print("Downloading CIFAR-10...")
    cifar_folder = DATASET_FOLDER / 'cifar10'
    cifar_folder.mkdir(parents=True, exist_ok=True)
    
    # Download training set
    datasets.CIFAR10(root=cifar_folder, train=True, download=True)
    # Download test set
    datasets.CIFAR10(root=cifar_folder, train=False, download=True)
    print("CIFAR-10 download completed!")


def download_fashion_mnist():
    """Download Fashion-MNIST dataset using torchvision"""
    print("Downloading Fashion-MNIST...")
    fmnist_folder = DATASET_FOLDER / 'fmnist'
    fmnist_folder.mkdir(parents=True, exist_ok=True)
    
    # Download training set
    datasets.FashionMNIST(root=fmnist_folder, train=True, download=True)
    # Download test set
    datasets.FashionMNIST(root=fmnist_folder, train=False, download=True)
    print("Fashion-MNIST download completed!")


def download_svhn():
    """Download SVHN dataset from official source"""
    print("Downloading SVHN...")
    svhn_folder = DATASET_FOLDER / 'svhn'
    svhn_folder.mkdir(parents=True, exist_ok=True)
    
    # URLs for SVHN dataset
    train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    
    train_path = svhn_folder / 'train_32x32.mat'
    test_path = svhn_folder / 'test_32x32.mat'
    
    # Download training data
    if not train_path.exists():
        print("Downloading SVHN training data...")
        urllib.request.urlretrieve(train_url, train_path)
        print("SVHN training data downloaded!")
    else:
        print("SVHN training data already exists.")
    
    # Download test data
    if not test_path.exists():
        print("Downloading SVHN test data...")
        urllib.request.urlretrieve(test_url, test_path)
        print("SVHN test data downloaded!")
    else:
        print("SVHN test data already exists.")
    
    # Verify the downloaded files can be loaded
    try:
        train_data = sio.loadmat(train_path)
        test_data = sio.loadmat(test_path)
        print(f"SVHN verification: Train shape: {train_data['X'].shape}, Test shape: {test_data['X'].shape}")
        print("SVHN download completed and verified!")
    except Exception as e:
        print(f"Error verifying SVHN data: {e}")


def download_imagenet32():
    """Download ImageNet-32 (Chrabaszcz et al. downsampled ImageNet 32x32) CIFAR-style batches."""
    print("Downloading ImageNet-32...")
    imagenet_folder = DATASET_FOLDER / "imagenet32"
    imagenet_folder.mkdir(parents=True, exist_ok=True)

    # Figshare article "Imagenet_32" (contains train_data_batch_1..10 and val_data)
    api_url = "https://api.figshare.com/v2/articles/4960082"

    # Fetch metadata (list of files + per-file download_url)
    print(f"Fetching file list from {api_url} ...")
    with urllib.request.urlopen(api_url) as r:
        meta = json.loads(r.read().decode("utf-8"))

    files = meta.get("files", [])
    if not files:
        raise RuntimeError(f"No files found at {api_url}. (API response missing 'files')")

    # Download each batch file by its direct download_url
    for f in files:
        name = f["name"]                      # e.g., "train_data_batch_1"
        url = f["download_url"]               # e.g., https://ndownloader.figshare.com/files/...
        expected_size = f.get("size", None)   # bytes (optional)
        dest = imagenet_folder / name

        if dest.exists():
            print(f"{name} already exists.")
            continue

        print(f"Downloading {name} ({expected_size} bytes) from {url} ...")
        urllib.request.urlretrieve(url, dest)

        # Basic verification: size check when available
        if expected_size is not None:
            got = dest.stat().st_size
            if got != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {name}: expected {expected_size} bytes, got {got} bytes. "
                    f"Delete {dest} and retry."
                )

    print("ImageNet-32 download completed!")

def main():
    """Download all datasets"""
    print(f"Dataset folder: {DATASET_FOLDER}")
    DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
    
    try:
        download_cifar10()
        download_fashion_mnist()
        download_svhn()
        download_imagenet32()
        print("\nAll datasets downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        raise


if __name__ == "__main__":
    main()
