import dtree
from typing import List, Tuple

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

def test_dtree(tree: dtree.DTree, fname: str)->Tuple[int, float]:
    with open(fname, "r") as fp:

        # skip name line
        line: str = fp.readline()
        while line:
            if line.strip() != '':
                line = fp.readline()
                break
            else:
                line = fp.readline()

        # check each param
        total: int = 0
        correct: int = 0
        while line:
            if line.strip != '':
                params_str: List[str] = line.split()
                params: List[int] = [int(x) for x in params_str]
                vector: dtree.Datapoint = dtree.Datapoint(params[:-1], params[-1])
                predicted_class: int = tree.classify_data(vector)
                if predicted_class == vector.classification:
                    correct += 1
                total += 1
            line = fp.readline()

        return total, correct / total
            
