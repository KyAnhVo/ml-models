# CS 4375: Introduction to Machine Learning (Honors)  
**Assignment 1, Part II: Decision Tree Induction**  

**Name:** [Your Full Name]  
**NetID:** [Your NetID]  
**Group Size:** 1 (individual submission)

---

## Environment
- Language: Python 3.6.9  
- Libraries:  
  - numpy 1.19.5  
  - pandas 1.1.5  
- No other non-standard libraries are used.  
- Code runs on UTD Linux machines with the above environment.

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

## Notes
- Tie-breaking rules, entropy calculation, and leaf class handling follow the assignment description.  
- Empty lines in data files are skipped.  
- **Learning Curve**: For part (d), the plot shown in the report was generated with a separate script using `matplotlib`. This script is **not part of the submission**, since only `numpy` and `pandas` are permitted in the implementation.  

---
