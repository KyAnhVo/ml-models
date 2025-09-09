import math
from typing import List, Optional, Tuple

class Datapoint:
    params: List[int]
    classification: int
    def __init__(self, params: Optional[List[int]] = None, classification: int = -1):
        self.params = params if params is not None else []
        self.classification = classification

class Node:
    l:                  int
    r:                  int
    remaining_params:   List[int]
    children:           'List[Optional[Node]]'
    def __init__(self, 
                 l: int = 0, 
                 r: int = 0, 
                 remaining_params: Optional[List[int]] = None, 
                 parent: 'Optional[Node]' = None, 
                 ):
        self.l, self.r = l, r
        self.remaining_params = remaining_params if remaining_params else []
        self.parent = parent
        self.children = [None, None, None]

class DTree:
    root:           Optional[Node]
    param_count:    int
    param_names:    List[str]
    dataset:        List[Datapoint]
    dataset_size:   int

    def __init__(self):
        self.root = None
        self.param_count = 0
        self.param_names = []
        self.dataset = []
        self.dataset_size = 0

    def classify_node(self, node: Node)->int:
        '''
        Classify a node. If a node is a true leaf then this function will choose
        its class. Else it will return -1. i.e. this can work as a de-facto
        is_pure(node) function with `if dtree.classify_node(curr_node) == -1`
        for example.

        Args:
            node (Node): a node that should be active
        Returns:
            - classification label if node is leaf
            - -1 otherwise
        '''
        if len(node.remaining_params) == 0: # cant decision-ize anymore
            count = [0, 0, 0]
            for i in range(node.l, node.r + 1):
                count[self.dataset[i].classification] += 1
            if count[0] >= count[1] and count[0] >= count[2]:
                return 0
            elif count[1] >= count[2]:
                return 1
            else:
                return 2
        else:
            dominate_class = self.dataset[node.l].classification
            for i in range(node.l + 1, node.r + 1):
                if dominate_class != self.dataset[i].classification:
                    return -1
            return dominate_class

    def information_gain(self, node: Node, param: int) -> float:
        '''
        calculate the information gain of a potential node split based
        on the given param. Also checks if param is in node.remaining_params
        first before moving on.

        Args:
            node (Node): node waiting to be splitted
            param (int): param to split node
        Returns:
            if valid then return potential IG gain, else -1.0f.
        '''
        #TODO: complete IG function

        if param not in node.remaining_params:
            return -1

        return 0
    
    def partition_node(self, node: Node, param: int) -> Tuple[int, int]:
        '''
        partitions the subarray from left to right to partitions which have
        param = 0 (1st part), param = 1 (2nd part), param = 2 (3rd part).
        Returns the tuple (b2, b3) containing the starting point of 2nd and 3rd parts.
        i.e. the partitionss are: `[node.l, b2 - 1]`, `[b2, b3 - 1]`, `[b3, r]`.

        Args:
            node (Node): a valid node hich is pointing to some correct subsection of
                array with its l, r attributes.
        Returns:
            b2, b3 (int, int): starting indices of partition 2 and partition 3, or
            (-1, -1) if error occurs.
        '''
        l = node.l
        r = node.r

        # From this part, essentially Dutch National Flag in subarray dataset[l, r]
        low, mid, high = l, l, r
        while mid <= high:
            v = self.dataset[mid].params[param]
            if v == 0:
                self.dataset[low], self.dataset[mid] = self.dataset[mid], self.dataset[low]
                low += 1
                mid += 1
            elif v == 1:
                mid += 1
            else:
                self.dataset[mid], self.dataset[high] = self.dataset[high], self.dataset[mid]
                high -= 1
        return low, mid

    def print_dataset(self):
        '''
        Print dataset of stored dataset in DTree object.
        '''
        line: str = ""
        for name in self.param_names:
            line += f"{name:>15}"
        tmp: str = "class"
        line += f"{tmp:>15}"
        print(line)

        for data in self.dataset:
            line = ""
            for val in data.params:
                line += f"{val:>15}"
            line+= f"{data.classification:>15}"
            print(line)


def entropy_term(p: float)->float:
    if p <= 0:
        return 0
    else:
        return -p * math.log2(p)


