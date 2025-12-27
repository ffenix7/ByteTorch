import numpy as np

class Datatype:
    def __init__(self, name: str, size_in_bytes: int, np_dtype):
        self.name = name
        self.size_in_bytes = size_in_bytes
        self.np_dtype = np_dtype

    def __repr__(self):
        return f"Datatype(name={self.name}, size_in_bytes={self.size_in_bytes})"

    def __eq__(self, other):
        return isinstance(other, Datatype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

# Predefined datatypes
INT32 = Datatype("int32", 4, np.int32)
FLOAT32 = Datatype("float32", 4, np.float32)
FLOAT64 = Datatype("float64", 8, np.float64)
BOOL = Datatype("bool", 1, np.bool_)