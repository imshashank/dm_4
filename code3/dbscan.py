from __future__ import division
import numpy as np
import time, csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as pl
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import math



X=[]
labels={}
docs_labels={}
print "loading docs"
'''
i = 0
with open('foo.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if i%100 == 0:
			print i
        X.append(list(row))
        i = i+1
       	if i>120:
       		break

'''

filename = 'graph_coord.pytext'
i =0
for line in open(filename):
		record = eval(line)
		temp =[]
		temp.append(record['x'])
		temp.append(record['y'])
		X.append(temp)
		#print i
		i=i+1


filename = 'label_id.pytext'
i=0
for line in open(filename):
		record = eval(line)
		docs_labels[record['id']] = record['lab_id']
		if labels.get(record['lab_id']) != None:
			labels[record['lab_id']] = labels.get(record['lab_id']) +1
		else:
			labels[record['lab_id']] = 1
		#if i > 1461:
		#	break
		i =i+1

filename = 'label_id.pytext'
i=0
for line in open(filename):
		record = eval(line)
		docs_labels[record['id']] = (record['lab_id']) 
		#if i > 1461:
		#	break
		i =i+1

X = np.array(X,dtype=float)
#print X
#print X.shape
print "docs loaded building cluster"
#labels_true = np.array(labels)
#print labels_true
#print labels_true.shape

#DBScan
batch_size = 45
#k = 10

#centers = [[50, 100], [100, 200], [50, 50],[150, 150]]
#centers=np.random.randint(size=(k,2),low=20,high=200)

#n_clusters = len(centers)

t0 = time.time()

#db = DBSCAN(eps=5, min_samples=5, metric='manhattan').fit(X)
X_scaled = StandardScaler().fit_transform(X)
t0 = time.time()
db = DBSCAN(eps=.04, min_samples=4, metric='euclidean', algorithm='auto').fit(X_scaled)

#k_means = KMeans(init='k-means++', n_clusters=n_clusters,verbose=True)
t_batch = time.time() - t0
#k_means.fit(X)
t_batch = time.time() - t0
#k_means_labels = k_means.labels_
db_labels = db.labels_
#copying example:
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_db = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_db)) - (1 if -1 in labels_db else 0)

#print('Estimated number of clusters: %d' % n_clusters_)
#print('Labels: ')
#print labels_db
##############################################################################
# Plot result


# Black removed and is used for noise instead.
unique_labels = set(labels_db)
'''
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k in (unique_labels):
	col = cm.spectral(float(k) / n_clusters_, 1)
	if k != -1:
		#col = 'k'
		# Black used for noise.
		
		class_member_mask = (labels_db == k)
		xy = X[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
'''
##Shashank's Entropy code:
#entropy

#n_clusters
#labels
#docs_labels

#print X.shape

#print labels_db

#rint labels_db.shape





ent = {}
size_cl={}
for k in range(n_clusters_):
	temp_cl = {}
	len_c = 0
	i = 0
	for x in labels_db:
		#cluster
		if x == k:
			l = docs_labels.get(i)
			if temp_cl.get(l) != None:
				temp_cl[l] = temp_cl.get(l) + 1
			else:
				temp_cl[l] = 1
			len_c =len_c  +1
		i = i+1
	#finding cluster entropy
	e = 0
	#print "cluster number "
	#print k
	size_cl[k]=len_c
	total = len_c
	#print "total docs in cluster"
	#print total
	for key, value in temp_cl.iteritems():
		val = value/total
		#print "Total"
		#print total
		#print "count of label"
		#print value
		
		#print value/total
		log_val= math.log(val,2)
		#print log_val
		#print val*log_val
		#print '\n'
		e = e + (val*log_val)
	e = e * -1
	#print "entropy for cluster"
	#print k
	#print e
	ent[k]=e



mean = total/n_clusters_
print "mean"
print mean
sum_v = 0
for key in size_cl:
	#print "size"
	#print size_cl.get(key)
	sum_v = sum_v + ((size_cl.get(key) - mean) ** 2)
	

skew= (math.sqrt(sum_v/n_clusters_))/mean

print "skew"
print skew

total = len(X)
print '\n'
#print "total"
#print total
fin_e=0

tot = 0
for keys in ent:
	#print keys
	tot = tot + size_cl.get(keys)
	#print size_cl.get(keys)
	fin_e = fin_e + (size_cl.get(keys)/total)*(ent.get(keys))
	
print "total docs"
print total
#print "valid points"
#print tot
print "final entropy"
print fin_e
print "time to cluster"
print t_batch
#k_means_cluster_centers = k_means.cluster_centers_
#k_means_labels_unique = np.unique(k_means_labels)


#mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,verbose=True)
#t0 = time.time()
#mbk.fit(X)
#t_mini_batch = time.time() - t0
#mbk_means_labels = mbk.labels_
#mbk_means_cluster_centers = mbk.cluster_centers_
#mbk _means_labels_unique = np.unique(mbk_means_labels)


