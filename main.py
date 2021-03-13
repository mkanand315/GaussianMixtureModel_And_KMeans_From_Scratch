import numpy
from kmeansV3 import KMeansModel

X = numpy.genfromtxt("Data.tsv", delimiter="\t")
    
K_Means = KMeansModel(X, 3)
K_Means.fit(X)

