import dtree
import data_reader
import sys

def main():
    tree: dtree.DTree = data_reader.construct_dtree(sys.argv[1])
    node: dtree.Node = dtree.Node(0, tree.dataset_size - 1, list(range(tree.dataset_size)), None)
    for i in range(tree.param_count):
        tree.partition_node(node, i)
        tree.print_dataset()
        print('------------------------------------------------------------------')

if __name__ == "__main__":
    main()
