import dtree
import data_reader
import sys

def main():
    tree: dtree.DTree = data_reader.construct_dtree(sys.argv[1])
    tree.print_dataset()

if __name__ == "__main__":
    main()
