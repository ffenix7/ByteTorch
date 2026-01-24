import queue
import threading
import random

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, prefetch_factor=2, collate_function=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.collate_function = collate_function if collate_function is not None else self._default_collate
        self.queue = queue.Queue(maxsize=prefetch_factor)
        
    def _worker(self, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            batch = self.collate_function(batch)
            self.queue.put(batch)
        self.queue.put(None)

    @staticmethod
    def _default_collate(batch):
        return batch

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        thread = threading.Thread(target=self._worker, args=(indices,))
        thread.start()
        
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch