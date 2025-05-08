#!/usr/bin/env python3

# Program to do link prediction using non-negative matrix factorization
#
# Mark Newman  23 MAY 2024

from sys import stdin
from random import random
from numpy import array,zeros,dot
from sklearn.decomposition import non_negative_factorization

CROSSV = 0.1
DIMEN = 80

# Read the data
def readdata():
    line = next(stdin)
    n0,n1,nedges = array(line.split(),int)
    B = zeros([n0,n1],int)
    removed = zeros([n0,n1],int)
    count = 0

    for k in range(n0):
        line = next(stdin)
        data = line.split()
        u = int(data[0])
        label = data[1]
        d = int(data[2])
        edge = array(data[4:4+d],int)
        for v in edge:
            if random()<CROSSV:
                removed[u,v] = 1
                count += 1
            else:
                B[u,v] = 1

    return n0,n1,nedges,count,B,removed


# Main program

# Read the data
n0,n1,nedges,nrem,B,removed = readdata()

print("Read network with",n0,"drugs and",n1,"diseases and",nedges,"edges")
print(nrem,"edges removed for cross-validation")

# Do the calculation
W,H,niter = non_negative_factorization(B,n_components=DIMEN)
predict = dot(W,H)

# Print out the results
filename = "nnmf.txt"
fp = open(filename,"w")
for u in range(n0):
    for v in range(n1):
        if B[u,v]==0:
            print(u,v,"{:.6g}".format(predict[u,v]),removed[u,v],file=fp)
fp.close()

