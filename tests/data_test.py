import os
import tempfile
import shutil
import numpy as np
import pytest
from src.data.dataloader import DataLoader
from src.data.dataset import Dataset
from src.core.tensor import Tensor

def create_txt_file(path, values):
    with open(path, 'w') as f:
        for v in values:
            f.write(f"{v}\n")

def create_csv_file(path, array):
    import pandas as pd
    pd.DataFrame(array).to_csv(path, index=False)

def test_dataset_and_dataloader_txt():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create a .txt file with floats
        txt_path = os.path.join(tmpdir, 'data.txt')
        values = [1.0, 2.0, 3.0]
        create_txt_file(txt_path, values)
        dataset = Dataset(tmpdir)
        assert len(dataset) == 1
        assert np.allclose(dataset[0].data, np.array(values))
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batches = list(loader)
        assert len(batches) == 1
        assert np.allclose(batches[0][0].data, np.array(values))
    finally:
        shutil.rmtree(tmpdir)

def test_dataset_and_dataloader_csv():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create a .csv file with 2D data
        csv_path = os.path.join(tmpdir, 'data.csv')
        array = [[1,2],[3,4]]
        create_csv_file(csv_path, array)
        dataset = Dataset(tmpdir)
        assert len(dataset) == 1
        assert np.allclose(dataset[0].data, np.array(array))
        # Test DataLoader with collate function
        def collate_fn(batch):
            return [x.data for x in batch]
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_function=collate_fn)
        batches = list(loader)
        assert len(batches) == 1
        assert np.allclose(batches[0][0], np.array(array))
    finally:
        shutil.rmtree(tmpdir)
