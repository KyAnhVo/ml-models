import numpy as np

class Soft_Clustering:
    def __init__(self, k, data, dtype):
        self.n, self.k = len(data), k

        self.data       = np.array(data, dtype=dtype)
        self.means      = np.zeros(shape=(k,), dtype=dtype)
        self.priors     = np.zeros(shape=(k,), dtype=dtype)
        self.variances  = np.zeros(shape=(k,), dtype=dtype)

        self.class_prob = np.zeros(shape=(len(data), k), dtype=np.float32)
        self.initialize_params()

    def initialize_params(self):
        for index in range(len(self.data)):
            classification = index % self.k
            self.class_prob[index, classification] = 1

        self.maximize()

    def estimate(self):
        for index, item in enumerate(self.data):
            denominator = 0
            for k in range(self.k):
                denominator += self.priors[k] * self.normal(self.means[k], self.variances[k], item)
            for k in range(self.k):
                numerator = self.priors[k] * self.normal(self.means[k], self.variances[k], item)
                self.class_prob[index, k] = numerator / denominator

    def maximize(self):
        effective_data_size = np.sum(self.class_prob, axis=0)
        
        # variance
        sqrd_dist_from_mean = (self.data[:, None] - self.means[None, :]) ** 2
        effective_sqrd_dist_from_mean = self.class_prob * sqrd_dist_from_mean
        numerator = np.sum(effective_sqrd_dist_from_mean, axis=0)
        denominator = np.sum(self.class_prob, axis=0)
        self.variances = (numerator / denominator).astype(self.data.dtype)

        # priors
        self.priors = (effective_data_size / self.n).astype(self.data.dtype)
        
        # means
        self.means = ((self.data[None, :] @ self.class_prob)[0] / np.sum(self.class_prob, axis=0)).astype(self.data.dtype)


    def train(self):
        self.estimate()
        self.maximize()
        #self.estimate()

    def normal(self, mean, variance, data):
        linear_term = 1 / np.sqrt(2 * np.pi * variance)
        exponential_term = -((data - mean) ** 2) / (2 * variance)
        return np.float32(linear_term * np.exp(exponential_term))
