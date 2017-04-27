from scipy.io import arff
import pandas as pd
from sklearn.metrics import silhouette_score

# FileNames saved from weka after doing the clustering
fileName = ['data/cluster2.arff', 'data/cluster3.arff', 'data/cluster5.arff']
# Number of clusters we will be needing
numberOfClusters = [2, 3, 5]
# Starting iteration on filenames
for i in range(0, len(fileName)):
    # Loading arff file
    data, meta = arff.loadarff(fileName[i])
    # Making a dataframe
    dataset = pd.DataFrame(data)
    # Defining labels
    labels = dataset["Cluster"]
    # Making final matrix after removing the not needed columns
    finalMatrix = dataset.drop(dataset.columns[[0,5,6]], axis=1)
    #
    # Calculating average silhouette for the clusters where K = 2, 3, 5
    #
    # The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean
    # nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b).
    # To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of.
    # Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
    #
    # Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    #
    averageSilhouette = silhouette_score(finalMatrix, labels, metric='euclidean')
    # Printing the average silhouette
    print "Average Silhouette for K = ",numberOfClusters[i]
    print "Average Silhouette: ", averageSilhouette
    print "================================================="


