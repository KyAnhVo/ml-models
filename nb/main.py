from typing import List

class NB:
    def __init__(self, x: List[List[int]], y: List[int], x_domain: List[int], y_domain: int):
        self.dataset_size = len(y)
        assert len(x) == self.dataset_size

        self.param_count = len(x[0])
        for vector in x:
            assert len(vector) == self.param_count

        self.x = x
        self.y = y
        self.x_domain = x_domain
        self.y_domain = y_domain
