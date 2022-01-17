import os
import sys
from pathlib import Path
import shutil

source_path = "./source/cats"
zip_path = source_path + "/cat-dataset.zip"



Path(source_path).mkdir(parents=True, exist_ok=True)
os.environ['KAGGLE_USERNAME'] = sys.argv[1]
os.environ['KAGGLE_KEY'] = sys.argv[2]

import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('crawford/cat-dataset', path=source_path, unzip=True,quiet=False,force=True)

if(os.path.exists(source_path+"/cats")):
    shutil.rmtree(source_path+"/cats",ignore_errors=True)

if(os.path.exists(zip_path)):
    os.remove(zip_path)
