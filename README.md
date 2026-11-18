# Document clustering coursework archive

[Shashank Agarwal](https://www.linkedin.com/in/shashankagarwal2/) and Anurag Kalra compared K-means, MiniBatchKMeans and DBSCAN in an Ohio State data mining assignment. The original notes credit Agarwal with K-means and Kalra with DBSCAN. [Original project and contribution notes](code3/Readme.md)

The experiment took feature vectors from an earlier assignment, computed pairwise Manhattan and Euclidean distances, and embedded the resulting matrices with multidimensional scaling. The clustering programs operate on those coordinates. The report compares runtime and uses topic-label entropy to describe the resulting groups.

| Start here | What it contains |
| --- | --- |
| [Lab report](report3/DataMininglab3%20%282%29.pdf) | The historical procedure and reported experiments |
| [Distance calculation](code3/distance.py) and [embedding](code3/distancetograph.py) | Feature vectors, pairwise distances and embedded coordinates |
| [K-means experiments](code3/kmeans.py) and [DBSCAN experiments](code3/dbscan.py) | The original clustering implementations and settings |

The report describes 14,302 documents with topic labels. A single dense matrix of that size contains 204,547,204 values. At eight bytes per value, it would use about 1.52 GiB before any copies, temporary arrays or CSV overhead. The historical submission omitted the large intermediate CSV. Start with a small subset if you want to explore the workflow on a modern machine, and check the original scripts for file and dependency assumptions.

For a fresh comparison, record the input set and embedding settings alongside the clustering results. Compare runtime and the percentage of documents labeled as noise alongside topic-label agreement. Entropy and agreement describe different properties, especially when clusters differ widely in size. [Scikit-learn clustering evaluation guide](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
