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
    splitting_param:    Optional[int]
    children:           'List[Optional[Node]]'
    classification:     Optional[int]
    def __init__(self, l: int = 0, r: int = 0, parent: 'Optional[Node]' = None):
        self.l, self.r = l, r
        self.parent = parent
        self.splitting_param = None
        self.children = [None, None, None]
        self.classification = None

class DTree:
    root:           Optional[Node]
    param_count:    int
    param_names:    List[str]
    dataset:        List[Datapoint]
    dataset_size:   int
    global_majority: int

    def __init__(self):
        self.root = None
        self.param_count = 0
        self.param_names = []
        self.dataset = []
        self.dataset_size = 0
        self.global_majority = -1

    def print_tree(self):
        assert self.root is not None
        self.print_tree_recursive(self.root, 0)
    
    def print_tree_recursive(self, node: Node, level: int):
        for i, child in enumerate(node.children):
            # Print indentation
            prefix = "| " * level
        
            # Print the attribute = value part
            assert node.splitting_param is not None
            line = f"{prefix}{self.param_names[node.splitting_param]} = {i} :"
            
            if child is None:
                if self.global_majority == -1:
                    global_class: List[int] = [0, 0, 0]
                    for i in range(self.dataset_size):
                        global_class[self.dataset[i].classification] += 1
                    self.global_majority = max(range(3), key= lambda x: global_class[x])
                line += f" {self.global_majority}"
                print(line)

            elif child.classification is not None:
                # Leaf node - print classification on same line
                line += f" {child.classification}"
                print(line)
            else:
                # Internal node - print line then recurse
                print(line)
                self.print_tree_recursive(child, level + 1)
    
    def print_tree_debug_pure(self, node: Node, level: int, available_params: List[int]):
        line: str = "|-" * level
        classification = node.classification
        if classification is not None:
            line += f"classification: {classification}"
            print(line)
        else:
            assert node.splitting_param is not None
            line += f"splitting param: {self.param_names[node.splitting_param]}"
            print(line)
            for param in available_params:
                line = "| " * level + "  +"
                ig: float = self.information_gain(node, param, available_params)
                line += f"{self.param_names[param]:<10}: {ig}"
                print(line)
            children_available_params = available_params.copy()
            children_available_params.remove(node.splitting_param)
            for child in node.children:
                if child is not None:
                    self.print_tree_debug_pure(child, level + 1, children_available_params)



    def classify_data(self, data: Datapoint):
        assert self.root is not None
        return self.classify_data_recursive(self.root, data)

    def classify_data_recursive(self, node: Node, data: Datapoint):
        if node.classification != -1:
            return node.classification
        else:
            assert node.splitting_param is not None
            param: int = node.splitting_param
            param_val: int = data.params[param]
            if node.children[param_val] is not None:
                child: Optional[Node] = node.children[param_val]
                assert child is not None
                return self.classify_data_recursive(child, data)
            else:
                if self.global_majority == -1:
                    global_class: List[int] = [0, 0, 0]
                    for i in range(self.dataset_size):
                        global_class[self.dataset[i].classification] += 1
                    self.global_majority = max(range(3), key= lambda x: global_class[x])

                return self.global_majority



    def build_tree(self)->None:
        ''' Build decision tree'''
        self.root = self.build_tree_recursive(l= 0, r= self.dataset_size - 1, remaining_params= [i for i in range(self.param_count)])

    def build_tree_recursive(self, l: int, r: int, remaining_params: List[int])->Node:
        '''
        Build tree recursively for a selected range and remaining params
        '''
        curr_node: Node = Node(l= l, r= r)
        classification = self.classify_node(curr_node, remaining_params)

        # if node classifiable (pure or no more remaining params), return
        if classification != -1:
            curr_node.classification = classification
            return curr_node
        
        # find param with max IG to decision-ize more, then partition array
        assert remaining_params is not None
        chosen_param: int = max(remaining_params, key= lambda x: self.information_gain(curr_node, x, remaining_params))
        curr_node.splitting_param = chosen_param
        start_1: int
        start_2: int
        start_1, start_2 = self.partition_node(curr_node, chosen_param)
        children_remaining_params = remaining_params.copy()
        children_remaining_params.remove(chosen_param)
        boundaries: List[Tuple[int, int]] = [(l, start_1 - 1), (start_1, start_2 - 1), (start_2, r)]

        # iterate over 3 boundaries, check if child is available then return.
        for i in range(3):
            l_child, r_child = boundaries[i]
            if l_child > r_child:
                continue
            else:
                child: Node = self.build_tree_recursive(l_child, r_child, children_remaining_params)
                child.parent = curr_node
                curr_node.children[i] = child

        # done
        return curr_node

    def classify_node(self, node: Node, remaining_params: List[int])->int:
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
        if len(remaining_params) == 0: # cant decision-ize anymore
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

    def information_gain(self, node: Node, param: int, remaining_params: List[int]) -> float:
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

        if param not in remaining_params:
            return -1

        h_node: float = 0
        h_children: List[float] = [0, 0, 0]

        # calculate h_node
        node_distribution: List[int] = [0, 0, 0]
        node_total: int = node.r - node.l + 1
        for i in range(node.l, node.r + 1):
            node_distribution[self.dataset[i].classification] += 1
        for i in range(0, 3):
            h_node += entropy_term(node_distribution[i] / node_total)

        # calculate children entropies
        children_distribution: List[List[int]] = [[0 for _ in range(3)] for _ in range(3)]
        children_total: List[int] = [0 for _ in range(3)]
        for i in range(node.l, node.r + 1):
            datapoint = self.dataset[i]
            current_child = datapoint.params[param]
            children_distribution[current_child][datapoint.classification] += 1
            children_total[current_child] += 1
        for child in range(3):
            if children_total[child] == 0:
                continue
            for classification in range(3):
                h_children[child] += entropy_term(children_distribution[child][classification] / children_total[child])
        for child in range(3):
            h_children[child] *= children_total[child] / sum(children_total)

        return h_node - sum(h_children)
    
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


