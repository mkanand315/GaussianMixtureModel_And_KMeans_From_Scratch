import numpy
import math

def Expectation_Maximization(X, K, epsilon):
    # # Dividing X into 3 subparts (each array has 50 elements)
    subparts = numpy.array_split(X, K)
    # # Computing the mean vector. It will have 3 elements, one mean for each subpart
    means = [numpy.mean(sub_arr, axis=0) for sub_arr in subparts]
    # # Computing covariances for each of the 3 subparts
    sigma = [numpy.cov(sub_arr.T) for sub_arr in subparts]
    # # Initializing pi vector with 3 equal probabilities as per assignment description
    pi =  [0.3333, 0.3333, 0.3333]

    old_log_likelihood = compute_log_likelihood(X, K, means, sigma, pi)
    
    # since GMM converges in 20-30 iterations
    for i in range(50):
        resp_matrix = ExpectationStep(X, K, means, sigma, pi)
        means, sigma, pi = MaximizationStep(X, K, resp_matrix)
        
        curr_log_likelihood = compute_log_likelihood(X, K, means, sigma, pi)
        if (abs(curr_log_likelihood - old_log_likelihood) < epsilon):
            break

        optimal_log_likelihood = curr_log_likelihood
    
    return resp_matrix

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
            resp_matrix[i][j] = pi[j]*compute_gaussian(X[i], means[j], sigma[j])/compute_likelihood(X[i], K, means, sigma, pi)
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

def predict_labels(K, resp_matrix):
    pred_labels = numpy.zeros(len(resp_matrix), dtype=int)

    for i in range(len(resp_matrix)):
        prediction = 0
        for k in range(K):
            if (resp_matrix[i][k] > resp_matrix[i][prediction]):
                prediction = k
        pred_labels[i] = prediction
    return pred_labels

def compute_gaussian(datapoint, mean_j, sigma_j):
    num_of_features = len(datapoint)
    acc = (2*numpy.pi)**num_of_features
    acc = acc * (numpy.linalg.det(sigma_j))
    acc = 1.0/(numpy.sqrt(acc))
    x_minus_means = numpy.matrix(datapoint - mean_j)
    gaussian = (acc)*numpy.exp(-0.5*(x_minus_means)*numpy.linalg.inv(sigma_j)*x_minus_means.T)
    return gaussian

def main():
    X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    resp_matrix = Expectation_Maximization(X, 3, 1e-5)
    pred_labels = predict_labels(3, resp_matrix)
    numpy.savetxt("gmm_output.tsv", pred_labels, delimiter="\t")

if __name__ == '__main__':
    main()