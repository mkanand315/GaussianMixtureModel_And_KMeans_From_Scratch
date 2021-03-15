import numpy

"""
Citations:
- Referenced helpful tutorial for numpy and linear algebra with
centroid matrix https://www.youtube.com/watch?v=W4fSRHeafMo
- Referenced for more insight on stopping criterion 
for K-Means https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
- Referenced for more insight on python's enemurate() function: https://www.programiz.com/python-programming/methods/built-in/enumerate
"""

class KMeansModel():

    """
    Initialization function for the model parameters.
    Based on the assignment description, we will have 150 data points 
    with 4 features each with the modified iris dataset: https://archive.ics.uci.edu/ml/index.php.
    """
    def __init__(self, X, num_of_k_clusters):
        self.num_of_samples, self.num_of_features = X.shape             # X.shape returns a tuple (rows, columns)
        self.K = num_of_k_clusters
        
    """
    This function goes through each point in the dataset and determines which
    centroid is the closest to that point. Then, it assigns the point to the 
    closest cluster by appending it to the list of that cluster
    """
    def make_and_assign_to_clusters(self, X, centroids):
        # Creating a list of list for clusters. One array for each cluster within a single list.
        clusters = [[] for _ in range(self.K)]

        # Looping through each point in training data to assign it to the closest cluster.
        for i, x_in_dataset in enumerate(X):
            # Determining which centroid is closest to the current point x in the dataset using Euclidean Distance.
            closest_cluster_centroid = numpy.argmin(numpy.sqrt(numpy.sum((x_in_dataset - centroids)**2, axis=1)))
            # Assigning training point to the closest cluster found in above step.
            clusters[closest_cluster_centroid].append(i)    
        return clusters
    
    """
    This functions updates the cluster centroid values by taking the average of each cluster.
    """
    def update_centroids(self, clusters, X):
        # Initializing array for new set of cluster centroids
        centroids = numpy.zeros((self.K, self.num_of_features))
        
        # Looping through each cluster to determine the mean of it along each row.
        for i, cluster in enumerate(clusters):
            # Computing the mean of the current cluster interation.
            updated_centroid = numpy.mean(X[cluster], axis=0)           # X[cluster] indexs all dataset points associated with the current cluster
            centroids[i] = updated_centroid
        return centroids

    """
    This function assigns a label to a datapoint based on the 
    cluster it belongs to
    """
    def predict_label_of_cluster(self, clusters, X):
        # Initializing array for label predictions
        pred_labels =  numpy.zeros(self.num_of_samples)

        # Looping through each cluster and points within each cluster to extract label predictions
        for i, cluster in enumerate(clusters):
            for j in cluster:
                # Assigning prediction label as label = cluster = 1, 2, ... K
                pred_labels[j] = i
        return pred_labels

    """
    This functions makes use of all the above functions to improve the centroids
    and cluster values
    """
    def fit(self, X):
        # Initial guess for centroids as given in assignment description
        centroids = numpy.array([[1.03800476, 0.09821729, 1.0469454, 1.58046376],
                    [0.18982966, -1.97355361, 0.70592084, 0.3957741],
                    [1.2803405, 0.09821729, 0.76275827, 1.44883158]])

        # Looping and updating centroids until not change in centroids occurs
        stop_flag = 0
        while stop_flag == 0:
            clusters = self.make_and_assign_to_clusters(X, centroids)
            
            # Saving cetroids of previous iteration for later.
            centroids_temp = centroids
            centroids =  self.update_centroids(clusters, X)
            
            # Calculating difference between old and new centroids for stopping criterion.
            assignment_stabilization_flag = numpy.sum(centroids - centroids_temp)

            # Stopping criterion. If (new) assignments stabilized, break out of loop
            if assignment_stabilization_flag == 0:
                print("centroids/assignments stabilized. stopping criterion met.")
                stop_flag = 1
 
        # Obtaining label predictions
        pred_labels = self.predict_label_of_cluster(clusters, X)
        # Saving predicted labels based on cluster assignments as output to "kmeans_output.tsv"
        numpy.savetxt("kmeans_output.tsv", pred_labels, delimiter="\t")
        
        

if __name__ == '__main__':
    X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    K_Means = KMeansModel(X, 3)
    K_Means.fit(X)