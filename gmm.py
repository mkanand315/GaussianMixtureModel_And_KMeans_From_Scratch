import numpy
import math

"""
Citations:
- Helpful resource in understanding performance differences between KMeans and GMM: 
https://towardsdatascience.com/gaussian-mixture-models-vs-k-means-which-one-to-choose-62f2736025f0
- Referenced for more insight on visual representation of GMM and responsibility matricies: 
http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html
- Referenced for numpy realted calculations for mean and covariances: 
https://towardsdatascience.com/gaussian-mixture-models-implemented-from-scratch-1857e40ea566
- Referenced for EM algorithm details: https://www.geeksforgeeks.org/gaussian-mixture-model/
"""

def Expectation_Maximization(X, K, epsilon):
    # Dividing X into 3 subparts (each array has 50 elements)
    subparts = numpy.array_split(X, K)
    # Computing the mean vector. It will have 3 elements, one mean for each subpart
    means = [numpy.mean(sub_arr, axis=0) for sub_arr in subparts]
    # Computing covariances for each of the 3 subparts
    sigma = [numpy.cov(sub_arr.T) for sub_arr in subparts]
    # Initializing pi vector with 3 equal probabilities as per assignment description. In general, it would be 1/K.
    pi =  [0.3333, 0.3333, 0.3333]

    # First log_likelihood with initialized parameters. This will be updated at each iteration
    old_log_likelihood = compute_log_likelihood(X, K, means, sigma, pi)
    
    # Looping 50 times since GMM usually converges in 20-30 iterations
    """
    Note: 
    With max_iteration of 20, i'm getting an accuracy of 78.66 (ie. not converged)
    With max_iteration of 25, i'm getting an accuracy of 79.33 (ie. not converged)
    With max_iteration of 30, i'm getting an accuracy of 80.66 (ie. not converged)
    With max_iteration of 50, i'm getting an accuracy of 82.0 
    With max_iteration of 100, i'm getting an accuracy of 82.0 
    """
    for i in range(50):
        # Computing the responsibility matrix for the current iteration. It determins how like a point is to belong to a given cluster
        resp_matrix = ExpectationStep(X, K, means, sigma, pi)

        # Updating the mean, covariance and pi parameters based on computed responsibility matrix
        means, sigma, pi = MaximizationStep(X, K, resp_matrix)
        
        # Computing the log_likelihood of current iteration
        curr_log_likelihood = compute_log_likelihood(X, K, means, sigma, pi)

        # Stopping criterion. If difference in log likelihood between the iterations is less than 1e^-5 then we break and have converged
        if (abs(curr_log_likelihood - old_log_likelihood) < epsilon):
            break

        old_log_likelihood = curr_log_likelihood
    
    # Predicting the labels based on the computed responsibility matrix
    pred_labels = numpy.zeros(len(resp_matrix), dtype=int)

    for i in range(len(resp_matrix)):
        prediction = 0
        for k in range(K):
            # This steps ensures that I assign the point the cluster which gives highest probability.
            # Reference: https://datascience.stackexchange.com/questions/14435/how-to-get-the-probability-of-belonging-to-clusters-for-k-means
            if (resp_matrix[i][k] > resp_matrix[i][prediction]):
                prediction = k
        pred_labels[i] = prediction
    return pred_labels

def compute_log_likelihood(X, K, means, sigma, pi):
    computed_log_likelihood = 0.0
    for i in range (len(X)):
        computed_log_likelihood = computed_log_likelihood + numpy.log(compute_likelihood(X[i], K, means, sigma, pi))
    return computed_log_likelihood 

def ExpectationStep(X, K, means, sigma, pi):
    num_of_samples, num_of_features = X.shape 
    resp_matrix = numpy.zeros((num_of_samples, K))

    # Computing the responsibility matrix here
    for i in range(num_of_samples):
        for j in range(K):
            numerator = pi[j]*compute_gaussian(X[i], means[j], sigma[j])
            denominator = compute_likelihood(X[i], K, means, sigma, pi)
            resp_matrix[i][j] = numerator / denominator
    return resp_matrix

def compute_likelihood(datapoint, K, means, sigma, pi):
    probability = 0.0
    for i in range(K):
        probability = probability + pi[i]*compute_gaussian(datapoint, means[i], sigma[i])
    return probability

def MaximizationStep(X, K, resp_matrix):
    num_of_samples, num_of_features = X.shape 

    new_means = numpy.zeros((K, num_of_features))
    new_sigma = numpy.zeros((K, num_of_features, num_of_features))
    pi = numpy.zeros(K)

    mariginal_resp_probabilities = numpy.zeros(K)

    for k in range(K):
        for i in range(num_of_samples):
            mariginal_resp_probabilities[k] = mariginal_resp_probabilities[k] + resp_matrix[i][k]
            new_means[k] = new_means[k] + (resp_matrix[i][k]) * X[i]
        new_means[k] = new_means[k] / mariginal_resp_probabilities[k]

        for i in range(num_of_samples):
            x_minus_means = numpy.zeros((1,num_of_features)) + X[i] - new_means[k]
            new_sigma[k] = new_sigma[k] + ( resp_matrix[i][k] / mariginal_resp_probabilities[k]) * x_minus_means * x_minus_means.T

        pi[k] = mariginal_resp_probabilities[k]/num_of_samples        
    
    return new_means, new_sigma, pi

def compute_gaussian(datapoint, mean_j, sigma_j):
    num_of_features = len(datapoint)
    acc = (2*numpy.pi)**(num_of_features/2)
    acc = acc * (numpy.linalg.det(sigma_j))**(-0.5)
    # acc = 1.0/(numpy.sqrt(acc))
    x_minus_means = numpy.matrix(datapoint - mean_j)
    gaussian = (acc)*numpy.exp(-0.5*(x_minus_means)*numpy.linalg.inv(sigma_j)*x_minus_means.T)
    return gaussian

def main():
    X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    pred_labels = Expectation_Maximization(X, 3, 1e-5)
    numpy.savetxt("gmm_output.tsv", pred_labels, delimiter="\t")

if __name__ == '__main__':
    main()