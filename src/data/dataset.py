from ..core.tensor import Tensor
import pathlib

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = pathlib.Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}!")
        
        files = sorted(self.dataset_path.rglob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in {self.dataset_path}")
        self.items = []

        for file in files:
            values = []
            with file.open() as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        raise ValueError(f"Empty line at {lineno}!")
                    try:
                        values.append(float(line))
                    except ValueError as e:
                        raise ValueError(
                            f"Failed to parse float in file {file}, line {lineno}: '{line}'"
                        ) from e
            
            self.items.append(Tensor(values))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Dataset(num_items={len(self.items)}, path='{self.dataset_path}')"
