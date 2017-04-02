import numpy as np
import sys
from DecisionTree import DecisionTree

categorical = [1,3,5,6,7,8,9,13]

def getFeature(samples, num):
    return np.array(samples[:, num])

def getFeatureValue(sample, i):
    return sample[i]

class RandomForest(DecisionTree):
    m = None
    def __init__(self, max_depth, m):
        self.max_depth = max_depth
        self.m = m

    def segmenter(self, S):
        #check categorical features vs quantitative features
        #if quantitative - sort vals, update entropy easily
        minH = sys.maxsize
        n = len(S)
        j = None
        B = None
        ftarray = [i for i in range(self.NUM_FEATURES)]
        ftarray = np.random.choice(ftarray, self.m, replace=False)
        for feature in ftarray:
            feature_vector = np.unique(getFeature(S, feature))
            feature_vector = np.sort(feature_vector)
            if len(feature_vector) > 1000:
                feature_vector = [feature_vector[i] for i in np.arange(0,len(feature_vector), 100)]
            if feature not in categorical:
                #initialize all points to right set
                H, split = self.quant_feature(feature, feature_vector, S, n)
            else: #categorical - try each split
                H, split = self.cat_feature(feature, feature_vector, S, n)
            if H < minH:
                minH = H
                j = feature
                B = split
                # print(minH)
                # print(j, B)
        return j, B
