import dtree
from typing import List

def construct_dtree(fname: str)->dtree.DTree:
    decision_tree: dtree.DTree = dtree.DTree()
    with open(fname, "r") as fp:
        line: str = fp.readline()

        # read class names first
        while line:
            if line.strip() != '':
                param_names: List[str] = line.split()[:-1]
                decision_tree.param_names = param_names
                decision_tree.param_count = len(param_names)
                line = fp.readline()
                break
            else: # skip empty lines
                line = fp.readline()

        # read data in
        while line:
            if line.strip() != '':
                vals = list(map(int, line.split()))
                params: List[int] = vals[:-1]
                classification: int = vals[-1]
                decision_tree.dataset.append(dtree.Datapoint(params, classification))
            line = fp.readline()
    
    decision_tree.dataset_size = len(decision_tree.dataset)
    return decision_tree
