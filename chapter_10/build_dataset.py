#
#  file:  build_dataset.py
#
#  Build the small MNIST dataset.
#
#  RTK, 03-Feb-2021
#  Last update:  24-Mar-2024
#
################################################################

import cv2
import numpy as np
import urllib.request
import gzip
import os


def is_valid_gzip(filepath):
    """Check if a file is a valid gzip file."""
    try:
        with gzip.open(filepath, 'rb') as f:
            f.read(1)  # Try to read one byte
        return True
    except (gzip.BadGzipFile, EOFError, OSError):
        return False


def download_mnist():
    """Download MNIST dataset from a reliable mirror."""
    # Using GitHub mirror as it's most reliable
    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    data = {}
    for key, filename in files.items():
        filepath = f"/tmp/{filename}"

        # Check if file exists and is valid, if not, remove and re-download
        if os.path.exists(filepath):
            if not is_valid_gzip(filepath):
                print(f"Corrupted file detected: {filename}. Removing and re-downloading...")
                os.remove(filepath)

        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                # Add headers to simulate browser request
                req = urllib.request.Request(
                    base_url + filename,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())

                # Verify the download was successful
                if not is_valid_gzip(filepath):
                    raise Exception(f"Downloaded file {filename} is not a valid gzip file")

            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise

        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                # Read image data
                f.read(16)  # Skip header
                buf = f.read()
                data[key] = np.frombuffer(buf, dtype=np.uint8)
                if 'train' in key:
                    data[key] = data[key].reshape(60000, 28, 28)
                else:
                    data[key] = data[key].reshape(10000, 28, 28)
            else:
                # Read label data
                f.read(8)  # Skip header
                buf = f.read()
                data[key] = np.frombuffer(buf, dtype=np.uint8)

    return data


def to_categorical(y, num_classes=10):
    """Convert labels to one-hot encoded format."""
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype='float32')
    categorical[np.arange(n), y] = 1
    return categorical


# Download MNIST data
mnist_data = download_mnist()
x_train = mnist_data['train_images']
y_train = mnist_data['train_labels']
x_test = mnist_data['test_images']
y_test = mnist_data['test_labels']

# Convert labels to categorical
ytrn = to_categorical(y_train)

# Create dataset directory if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Save full-size datasets
np.save("dataset/train_images_full.npy", x_train)
np.save("dataset/test_images_full.npy", x_test)
np.save("dataset/train_labels_vector.npy", ytrn)
np.save("dataset/train_labels.npy", y_train)
np.save("dataset/test_labels.npy", y_test)

# Build 14x14 versions
xtrn = np.zeros((60000, 14, 14), dtype="float32")
for i in range(60000):
    xtrn[i, :, :] = cv2.resize(x_train[i], (14, 14), interpolation=cv2.INTER_LINEAR)

xtst = np.zeros((10000, 14, 14), dtype="float32")
for i in range(10000):
    xtst[i, :, :] = cv2.resize(x_test[i], (14, 14), interpolation=cv2.INTER_LINEAR)

np.save("dataset/train_images_small.npy", xtrn)
np.save("dataset/test_images_small.npy", xtst)

print("Dataset built successfully!")
