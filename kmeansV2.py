import numpy

class KMeans():

    """
    Initialization function for the model parameters.
    Based on the assignment description, we will have 150 data points 
    with 4 features each.
    """
    def __init__(self, X, num_of_clusters):
        self.K = num_of_clusters
        self.num_of_samples, self.num_of_features = X.shape
        self.max_iters = 200


    """
    This function is equivalent to generating random, initial, centroids.
    Since we are given the initial centroid values, this function simple 
    returns the specified intial centroids.
    """
    def initialize_centroids(self, X):
        centroids = numpy.array([[1.03800476, 0.09821729, 1.0469454, 1.58046376],
                    [0.18982966, -1.97355361, 0.70592084, 0.3957741],
                    [1.2803405, 0.09821729, 0.76275827, 1.44883158]])
        return centroids

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
        centroids = self.initialize_centroids(X)

        for i in range(self.max_iters):
            clusters = self.create_clusters(X, centroids)
            
            centroids_temp = centroids
            centroids =  self.update_centroids(clusters, X)

            difference = centroids - centroids_temp

            if not difference.any():
                print("stopping criterion met")
                break

        pred_labels = self.predict_label_of_cluster(clusters, X)
        numpy.savetxt("kmeans_output.tsv", pred_labels, delimiter="\t")


# """
# Script to call the above functions
# """
# if __name__ == '__main__':
#     X = np.genfromtxt("Data.tsv", delimiter="\t")
    
#     Kmeans = KMeans(X, 3)
#     pred_labels = Kmeans.fit(X)

#     # Saving the predicted labels in 
#     numpy.savetxt("kmeans_output.tsv", pred_labels, delimiter="\t")
