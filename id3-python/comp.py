
import sys

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        l1, l2 = f1.readline().rstrip(), f2.readline().rstrip()

        while l1 and l2:
            if l1 != l2:
                print("Diff:")
                print(f"\tl1 = {l1}")
                print(f"\tl2 = {l2}")
            l1, l2 = f1.readline().rstrip(), f2.readline().rstrip()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <file1> <file2>")
        sys.exit(1)
    
    compare_files(sys.argv[1], sys.argv[2])
