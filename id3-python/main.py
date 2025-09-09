import dtree
import data_reader
import sys

def main():
    tree: dtree.DTree = data_reader.construct_dtree(sys.argv[1])
    print(tree.dataset_size, tree.param_count, tree.param_names)
    tree.build_tree()
    assert tree.root is not None
    tree.print_tree_debug_pure(tree.root, 0)

if __name__ == "__main__":
    main()
