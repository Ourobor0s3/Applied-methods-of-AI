# load_btc_info.py

import kagglehub
from constants import DATA_DIR

# Download latest version
path = kagglehub.dataset_download(
    "novandraanugrah/bitcoin-historical-datasets-2018-2024",
    output_dir=DATA_DIR)

print("Path to dataset files:", path)