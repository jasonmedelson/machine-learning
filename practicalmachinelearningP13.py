# tutorial video part 16
# Creating K nearest neighbors Algorithm
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

# euclidian_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
style.use("fivethirtyeight")

dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color = i)
# plt.scatter(new_features[0],new_features[1])
# plt.show()

def k_nearest_neighbors(data,predict, k=3):
    if len(data) >= k:
        warnings.warn('K is less than total groups')  #warn bad k value
        knnalgos
        return vote_results
