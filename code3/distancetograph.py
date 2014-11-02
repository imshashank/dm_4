import csv,sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
csv.field_size_limit(sys.maxsize)
# Distance file available from RMDS project:
#    https://github.com/cheind/rmds/blob/master/examples/european_city_distances.csv

np.set_printoptions(precision=2)
dists = []
cities = []
i = 0
print "something"
with open('foo.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        #print row
        dists.append(row)
        cities.append(i)
        i = i+1

'''
reader = csv.reader(open("foo.csv", "r"), delimiter=';')
data = list(reader)

print data

i=0
for d in data:
    cities.append(i)
    dists.append(map(float , d[1:-1]))
    i = i+1
'''
#print dists
#print dists

print "files loades now finding graph plot points"
adist = np.array(dists, dtype= float)
#print adist
amax = np.amax(adist)
adist /= amax

mds = manifold.MDS(n_components=2, dissimilarity="precomputed",n_jobs=-1)
results = mds.fit(adist)

coords = results.embedding_
print "Time to plot"
#plt.subplots_adjust(bottom = 0.1)

#plt.scatter(
#    coords[:, 0], coords[:, 1], marker = 'o'
#    )
graph_coord = open('graph_coord_2.pytext','w')
i=0
for label, x, y in zip(cities, coords[:, 0], coords[:, 1]):
    temp ={}
    temp['id']= i
    temp['x']= x
    temp['y']= y
    i = i + 1
    print >> graph_coord,temp
    '''
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    '''

#plt.show()