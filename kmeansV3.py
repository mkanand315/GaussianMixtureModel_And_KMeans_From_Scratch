import numpy

"""
Citations:
- Referenced helpful tutorial for numpy and linear algebra with
centroid matrix https://www.youtube.com/watch?v=W4fSRHeafMo
- Referenced for more insight on stopping criterion 
for K-Means https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
"""

class KMeansModel():

    """
    Initialization function for the model parameters.
    Based on the assignment description, we will have 150 data points 
    with 4 features each.
    """
    def __init__(self, X, num_of_clusters):
        self.K = num_of_clusters
        self.num_of_samples, self.num_of_features = X.shape

    """
    This function is equivalent to generating random, initial, centroids.
    Since we are given the initial centroid values, this function simple 
    returns the specified intial centroids.
    """

    """
    This function goes through each point in the dataset and determines which
    centroid is the closest to that point. Then, it assigns the point to the 
    closest cluster
    """
    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]

        for i, x_in_dataset in enumerate(X):
            # Determining which centroid is closest to the current point x in the dataset
            closest_centroid = numpy.argmin(numpy.sqrt(numpy.sum((x_in_dataset - centroids)**2, axis=1)))
            clusters[closest_centroid].append(i)    
        return clusters
    
    """
    This functions updates the centroid values by taking the average of the clusters.
    """
    def update_centroids(self, clusters, X):
        centroids = numpy.zeros((self.K, self.num_of_features))
        
        for i, cluster in enumerate(clusters):
            updated_centroid = numpy.mean(X[cluster], axis=0)
            centroids[i] = updated_centroid
        return centroids

    """
    This function assigns a label to a datapoint based on the 
    cluster it belongs to
    """
    def predict_label_of_cluster(self, clusters, X):
        pred_labels =  numpy.zeros(self.num_of_samples)

        for i, cluster in enumerate(clusters):
            for j in cluster:
                pred_labels[j] = i
        return pred_labels

    """
    This functions makes use of all the above functions to improve the centroids
    and cluster values
    """
    def fit(self, X):
        centroids = centroids = numpy.array([[1.03800476, 0.09821729, 1.0469454, 1.58046376],
                    [0.18982966, -1.97355361, 0.70592084, 0.3957741],
                    [1.2803405, 0.09821729, 0.76275827, 1.44883158]])

        stopFlag = 0
        while stopFlag == 0:
            clusters = self.create_clusters(X, centroids)
            
            centroids_temp = centroids
            centroids =  self.update_centroids(clusters, X)

            difference = numpy.sum(centroids - centroids_temp)

            if difference == 0:
                print("stopping criterion met")
                stopFlag = 1
 
        pred_labels = self.predict_label_of_cluster(clusters, X)
        numpy.savetxt("kmeans_output.tsv", pred_labels, delimiter="\t")