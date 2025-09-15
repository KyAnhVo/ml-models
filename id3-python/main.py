import dtree
import data_reader
import sys

def main():
    tree: dtree.DTree = data_reader.construct_dtree(sys.argv[1])
    tree.build_tree()
    assert tree.root is not None
    tree.print_tree()
    train_data_analytics    = data_reader.test_dtree(tree, sys.argv[1])
    test_data_analytics     = data_reader.test_dtree(tree, sys.argv[2])
    print(f'\nAccuracy on training set ({train_data_analytics[0]} instances): {train_data_analytics[1]*100:.1f}%')
    print(f'\nAccuracy on test set ({test_data_analytics[0]} instances): {test_data_analytics[1]*100:.1f}%')

if __name__ == "__main__":
    main()
