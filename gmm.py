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

"""
This function runs the EM algorithm. It makes use of the helper functions for the Expectation and Maximization steps as well
as gaussian, likelihood, and log-likelihood computations.
It returns an array of predicted labels after the EM algorithm has convereged or had it's max number of iterations.
"""
def Expectation_Maximization(X, KmeansOutput, K, epsilon):

    """
    Note: In order to obtain the right accruacy, as in the assignment description, I had to initialize the parameters
    using the output of the kmeans model and compute parameter intializations based on the clusters.

    The KMeansOutput parameter of this function stores the output of the KMeansModel
    """
    # Determining Model Parameters Initializations based on KMeansOutput from kmeans_output.tsv
    cluster1 = []
    cluster2 = []
    cluster3 = []

    for i in range(len(KmeansOutput)):
        if (KmeansOutput[i] == 0):
            cluster1.append(X[i])
        if(KmeansOutput[i] ==  1):
            cluster2.append(X[i])
        if(KmeansOutput[i] == 2):
            cluster3.append(X[i])

    subpart1 = numpy.array(cluster1)
    subpart2 = numpy.array(cluster2)
    subpart3 = numpy.array(cluster3)

    mean1 = numpy.mean(subpart1, axis=0)
    mean2 = numpy.mean(subpart2, axis=0)
    mean3 = numpy.mean(subpart3, axis=0)

    sigma1 = numpy.cov(subpart1.T)
    sigma2 = numpy.cov(subpart2.T)
    sigma3 = numpy.cov(subpart3.T)
    
    # Appending the means and covariances to create the intializations
    means = numpy.array([mean1, mean2, mean3])
    sigma = numpy.array([sigma1, sigma2, sigma3])
    pi =  [0.3333, 0.3333, 0.3333]

    # First log_likelihood with initialized parameters. This will be updated at each iteration
    old_log_likelihood = compute_log_likelihood(X, K, means, sigma, pi)
    
    """
    Note: When using the splitting technique as in the assignment description, I obtain the following: 
    With max_iteration of 20, i'm getting an accuracy of 78.66 (ie. not converged)
    With max_iteration of 25, i'm getting an accuracy of 79.33 (ie. not converged)
    With max_iteration of 30, i'm getting an accuracy of 80.66 (ie. not converged)
    With max_iteration of 50, i'm getting an accuracy of 82.0 
    With max_iteration of 100, i'm getting an accuracy of 82.0 

    When using the kmeans output to determine the intializations for means and covariances, I get accuracy of 96.67.
    """
    # Looping 50 times since GMM usually converges in 20-30 iterations to be safe
    for i in range(50):
        # Computing the responsibility matrix for the current iteration. It determins how like a point is to belong to a given cluster
        resp_matrix = ExpectationStep(X, K, means, sigma, pi)

        # Updating the mean, covariance and pi parameters based on computed responsibility matrix
        means, sigma, pi = MaximizationStep(X, K, resp_matrix)
        
        # Computing the log_likelihood of current iteration
        curr_log_likelihood = compute_log_likelihood(X, K, means, sigma, pi)

        # Stopping criterion. If difference in log likelihood between the iterations is less than 1e^-5 then we break and have converged
        if (abs(curr_log_likelihood - old_log_likelihood) < epsilon):
            print("Stopping criterion met. Breaking now")
            break

        old_log_likelihood = curr_log_likelihood
    
    # Predicting the labels based on the computed responsibility matrix
    pred_labels = numpy.zeros(len(resp_matrix), dtype=int)

    for i in range(len(resp_matrix)):
        prediction = 0
        for k in range(K):
            # This steps ensures that I assign the point to a cluster which gives highest probability.
            # Reference: https://datascience.stackexchange.com/questions/14435/how-to-get-the-probability-of-belonging-to-clusters-for-k-means
            if (resp_matrix[i][k] > resp_matrix[i][prediction]):
                prediction = k
        pred_labels[i] = prediction
    return pred_labels

"""
This function computes the log likelihood which is used to determine when the algorithm will converge
"""
def compute_log_likelihood(X, K, means, sigma, pi):
    computed_log_likelihood = 0.0
    for i in range (len(X)):
        computed_log_likelihood = computed_log_likelihood + numpy.log(compute_likelihood(X[i], K, means, sigma, pi))
    return computed_log_likelihood 

"""
This function carries out the Expectation step of the EM algorithm. It computes the responsibility matrix which determines
how likely a point is to belong to a cluster k.
"""
def ExpectationStep(X, K, means, sigma, pi):
    num_of_samples, num_of_features = X.shape 
    resp_matrix = numpy.zeros((num_of_samples, K))

    # Computing the responsibility matrix here for all points
    for i in range(num_of_samples):
        for j in range(K):
            # Following Equation (15.12) as in the notes
            numerator = pi[j]*compute_gaussian(X[i], means[j], sigma[j])
            denominator = compute_likelihood(X[i], K, means, sigma, pi)
            # Calculating the ij position of the responsibility matrix to determine how likely point x belongs to cluster j
            resp_matrix[i][j] = numerator / denominator
    return resp_matrix

def compute_likelihood(datapoint, K, means, sigma, pi):
    probability = 0.0
    for i in range(K):
        # As per Equation (15.12) in the notes
        probability = probability + pi[i]*compute_gaussian(datapoint, means[i], sigma[i])
    return probability

"""
This function carries out the maximization step of the EM algorithm. It improves the model parameters 
which are the mean, covariances and pi
"""
def MaximizationStep(X, K, resp_matrix):
    num_of_samples, num_of_features = X.shape 

    new_means = numpy.zeros((K, num_of_features))
    new_sigma = numpy.zeros((K, num_of_features, num_of_features))
    pi = numpy.zeros(K)

    revised_r_matrix = numpy.zeros(K)

    # Maximum Likelihood for the model parameters (mean, covariances, pi)
    for k in range(K):
        # Calculating the new mean vector as per Equation (15.16) in the notes
        for i in range(num_of_samples):
            revised_r_matrix[k] = revised_r_matrix[k] + resp_matrix[i][k]
            new_means[k] = new_means[k] + (resp_matrix[i][k]) * X[i]
        new_means[k] = new_means[k] / revised_r_matrix[k]

        # Calculating the new covariance matrix as per Equation (15.17) in the notes
        for i in range(num_of_samples):
            x_minus_means = numpy.zeros((1,num_of_features)) + X[i] - new_means[k]
            new_sigma[k] = new_sigma[k] + ( resp_matrix[i][k] / revised_r_matrix[k]) * x_minus_means * x_minus_means.T

        # Calculating new pi values as per Equation (15.18) of the notes
        pi[k] = revised_r_matrix[k]/num_of_samples        
    
    return new_means, new_sigma, pi

"""
This function computes the multivariate normal distribution
"""
def compute_gaussian(datapoint, mean_j, sigma_j):
    return (2*numpy.pi)**(-len(datapoint)/2)*numpy.linalg.det(sigma_j)**(-1/2)*numpy.exp(-numpy.dot(numpy.dot((datapoint-mean_j).T, numpy.linalg.inv(sigma_j)), (datapoint-mean_j))/2)

"""
This is the driver function to execute the model.
"""
def main():
    X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    # Using the output of Kmeans to compute initial parameters for gmm model.
    KMeansOutput = numpy.genfromtxt("kmeans_output.tsv", delimiter="\t")
    pred_labels = Expectation_Maximization(X, KMeansOutput, 3, 1e-5)
    numpy.savetxt("gmm_output.tsv", pred_labels, delimiter="\t")

if __name__ == '__main__':
    main()