import sys

if len(sys.argv) != 3:
    print("Usage: python compare.py file1 file2")
    sys.exit(1)

file1, file2 = sys.argv[1], sys.argv[2]

def read_nonempty_stripped(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line for line in f]

lines1 = read_nonempty_stripped(file1)
lines2 = read_nonempty_stripped(file2)

if lines1 == lines2:
    print("Files match (non-empty, stripped lines).")
    sys.exit(0)

print("Files differ:")
max_len = max(len(lines1), len(lines2))

for i in range(max_len):
    a = lines1[i] if i < len(lines1) else "<NO LINE>"
    b = lines2[i] if i < len(lines2) else "<NO LINE>"
    if a != b:
        print(f"Line {i+1}:")
        print(f" {file1}: {a}")
        print(f" {file2}: {b}")
