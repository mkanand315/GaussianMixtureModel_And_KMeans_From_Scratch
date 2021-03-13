import numpy
from kmeansV2 import KMeans

X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    
Kmeans = KMeans(X, 3)
Kmeans.fit(X)

