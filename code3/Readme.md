Data Mining Project for CSE OSU CLass
===========
By
Shashank Agarwal (agarwal.202@osu.edu)
Anurag Kalra (kalra.25@osu.edu)

all code is in the code3 folder and reports are in the report3 folder in pdf and docx format

##Instructions

We are using the feature vector made in the first assignment which is in the file 'feature_matrix.pytext'. 

Procedure:
1) We converted the feature vector in file "feature_matrix.pytext" to a distance matrix saved in file "foo.csv" for manhattan and euclidean.
As the file foo.csv was over 5GB we are not including that in the report. But the file can be generated executint the below file.
code_file: distance.py

2)Using the distance matrix we created the graph co-ordinates for manhattan and euclidean distances.
code_file: distancetographg.py
Output:
	graph_coord.pytext (Manhattan Distance coordinates)
	graph_coord_2.pytext  (Manhattan Distance coordinates)

3)We used the above graph coordinates for clustering.

3.1) KMeans:
	We are using two approaches
	1)k-Means 
	2)MiniBatchKMeans

The file 'kmeans.py' includes code for both algorithms and also plots the graphs for both. You can just pass the value of "k" in command line.
Example: python kmeans.py 10 #where 10 is the value of k
code_file: kmeans.py

3.2) DBScan
The code for dbscan is saved in file dbscan.py. The value of epsilon and min can be changed in line 91:
db = DBSCAN(eps=.04, min_samples=4, metric='euclidean', algorithm='auto').fit(X_scaled)
code_file: dbscan.py

===========================================================================================================================

Entropy:
For entropy all labels were converted to "integer" values. Entropy of wach cluster config is shown at execution of script.
Output_file: label_id.py

===========================================================================================================================
Executing the Code:

To run KMeans enter:
	make kmeans

#the default 'k' is 10 in makefile for other values run: 'python kmeans.py k' and change value of k as desired

To run DBScan please enter:
	make dbscan

===========================================================================================================================
Contributions:
Shashank Agarwal implemented the "kmeans" clustering
Anurag Kalra implemented the "dbscan" clustering

P.S. We had to remove the "foo.csv" (distance matrix) file as the folder exceeded the allowed size of submission






