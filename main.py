import numpy
from kmeansV3 import KMeans

X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    
Kmeans = KMeans(X, 3)
Kmeans.fit(X)

