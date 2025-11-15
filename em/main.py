#!/bin/python3

import sys
from soft_clusterer import Soft_Clustering

FILE    = sys.argv[1]
K       = int(sys.argv[2])
ITER    = int(sys.argv[3])

def main():
    data = []
    with open(FILE, mode='r') as f:
        for line in f:
            data.append(float(line))
    
    clusterer = Soft_Clustering(K, data)
    
    for i in range(ITER):
        print(f"After iteration {i + 1}:")
        for k in range(K):
            print(f"Gaussian {k + 1}: mean = {clusterer.means[k]:.4f}, variance = {clusterer.variances[k]:.4f}, prior = {clusterer.priors[k]:.4f}")
        print()
        clusterer.train()

if __name__ == "__main__":
    main()
