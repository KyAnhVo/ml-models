# CS 4375: Introduction to Machine Learning (Honors)  
**Assignment 1, Part II: Decision Tree Induction**  

**Name:** Ky Anh Vo
**NetID:** kxv220016
**Group Size:** 1 (individual submission)

---

## Environment
- Language: Python 3.6.9  
- Used pure Python, no non-standard libraries are used.

---

## How to Run
The entry point is `main.py`.  

```bash
python main.py <train_file> <test_file>
```

Example:
```bash
python main.py train.dat test.dat
```

---

## File Descriptions
- **main.py** → Entry point; parses arguments, trains the decision tree, evaluates accuracy.  
- **data_reader.py** → Reads and parses datasets (skips empty lines).  
- **dtree.py** → Implements the ID3 decision tree algorithm, including entropy calculation, tie-breaking, and classification.  

---

## Output
The program prints to `stdout`:  
1. The learned decision tree, in the specified nested text format.  
2. Accuracy on the training set (two decimal places).  
3. Accuracy on the test set (two decimal places).  

---

## Learning Curve comment
- The learning curve graph was plotted in matplotlib, with the general algorithm being construct data and each time remove 5% of dataset and test until 5%, repeated 50 times.
- As expected, the curves averaged out on a concaved non-decreasing graph, which plateus at about 87.x% at dataset size = 100%

___
