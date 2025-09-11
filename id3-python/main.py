import dtree
import data_reader
import sys

def main():
    tree: dtree.DTree = data_reader.construct_dtree(sys.argv[1])
    tree.build_tree()
    assert tree.root is not None
    tree.print_tree()

if __name__ == "__main__":
    main()
