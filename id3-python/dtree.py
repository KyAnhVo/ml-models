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
            -1 if not pure node, else classification of node.
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
    
    def partition_node(self, node: Node, param: int) -> Tuple[int, int]:
        '''
        partitions the subarray from left to right to partitions which have
        param = 0 (1st part), param = 1 (2nd part), param = 2 (3rd part).
        Returns the tuple (b2, b3) containing the starting point of 2nd and 3rd parts.
        i.e. the partitionss are: [node.l, b2 - 1], [b2, b3 - 1], [b3, r].

        Args:
            node (Node): a valid node hich is pointing to some correct subsection of
                array with its l, r attributes.
        Returns:
            b2, b3 (int, int): starting indices of partition 2 and partition 3, or
            (-1, -1) if error occurs.
        '''
        if node.l >= node.r:
            return (-1, -1)
        
        # From this part, essentially Dutch National Flag in subarray dataset[l, r]
        left, right, mid = node.l, node.l, node.r
        for i in range(node.l, node.r + 1):
            if i > left:
                break
            if self.dataset[i].params[param] == 0:
                # swap left and mid
                # then swap i and left
                self.dataset[left], self.dataset[mid] = self.dataset[mid], self.dataset[left]
                self.dataset[i], self.dataset[left] = self.dataset[left], self.dataset[i]
                left += 1
                mid += 1
            elif self.dataset[i].params[param] == 1:
                # swap i and mid
                self.dataset[mid], self.dataset[i] = self.dataset[i], self.dataset[mid]
                mid += 1
            else:
                self.dataset[right], self.dataset[i] = self.dataset[i], self.dataset[right]
                right -= 1
        return left, mid

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


