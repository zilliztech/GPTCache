import os

import faiss
from faiss import IndexHNSWFlat, Index
import numpy as np


class Faiss:
    index: Index

    def __init__(self, index_file_path, dimension):
        self.index_file_path = index_file_path
        self.index = IndexHNSWFlat(dimension, 32)
        if os.path.isfile(index_file_path):
            self.index = faiss.read_index(index_file_path)

    def add(self, data):
        np_data = np.array(data).astype('float32')
        self.index.add(np_data)

    def mult_add(self, datas):
        np_data = np.array(datas).astype('float32')
        self.index.add(np_data)

    def search(self, data):
        np_data = np.array(data).astype('float32')
        D, I = self.index.search(np_data, 1)
        distance = int(D[0, 0] * 100)
        vector_data = self.index.reconstruct(int(I[0, 0])).reshape(1, -1)
        return distance, vector_data

    def close(self):
        faiss.write_index(self.index, self.index_file_path)
