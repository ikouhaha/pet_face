import os
import sys
from pathlib import Path

source_path = "./source"
Path(source_path).mkdir(parents=True, exist_ok=True)
os.environ['KAGGLE_USERNAME'] = sys.argv[0]
os.environ['KAGGLE_KEY'] = sys.argv[1]

print(sys.argv)

import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('crawford/cat-dataset', path=source_path, unzip=True)