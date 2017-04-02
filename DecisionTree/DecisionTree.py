import numpy as np
import sys
from Node import Node, InNode, LeafNode
NUM_FEATURES = 12
LABEL_INDEX = 13
NUM_CLASSES = 2
categorical = []#[1,3,5,6,7,8,9,13]


def getFeature(samples, num):
    return np.array(samples[:, num])

def getFeatureValue(sample, i):
    return sample[i]

class DecisionTree:
    root = None
    LABEL = None
    NUM_FEATURES = None
    NUM_SAMPLES = None

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def getLabel(self, sample):
        return sample[self.LABEL]

    def entropy(self, S, samples): #samples is a list of ints
        counts = [0,0]
        l = len(samples)
        if l != 0:
            for i in samples:
                if self.getLabel(S[i]) == 0:
                    counts[0] += 1
                else:
                    counts[1] += 1
            total = 0
            for count in counts:
                pc = count/l
                if pc == 1:
                    return 0
                total += pc*np.log(pc)
            return -total
        else:
            return 0

    def split_entropy(self, S, left, right):
        lsize = len(left)
        rsize = len(right)
        lh = self.entropy(S, left)
        rh = self.entropy(S, right)
        total = lsize*lh + rsize*rh
        return total/(lsize+rsize)

    def getMajority(self, S):
        counts = [0,0]
        for sample in S:
            counts[self.getLabel(sample)] += 1
        if counts[0] > counts[1]:
            return 0
        return 1

    def quant_feature(self, feature, fvector, S, n):
        minH = sys.maxsize
        split = None
        left = set()
        right = set()
        for i in range(n):
            right.add(i)
        for split_value in fvector:
            for index in range(n):
                if S[index][feature] < split_value:
                    left.add(index)
            right.difference_update(left)
            H = self.split_entropy(S, left, right)
            if H < minH:
                minH = H
                split = split_value
        return (minH, split)

    def cat_feature(self, feature, fvector, S, n):
        minH = sys.maxsize
        split = None
        left = set()
        right = set()
        for split_value in fvector:
            for index in range(n):
                if S[index][feature] == split_value:
                    left.add(index)
                else:
                    right.add(index)
            H = self.split_entropy(S, left, right)
            if H < minH:
                minH = H
                split = split_value
            left.clear()
            right.clear()
        return (minH, split)

    def checkIfLeaf(self, S):
        if len(S) == 0:
            return False
        counts = [0,0]
        for sample in S:
            counts[sample[self.LABEL]] += 1
        for count in counts:
            if count/len(S) >= .9:
                return True
        return False

    def segmenter(self, S):
        minH = sys.maxsize
        n = len(S)
        j = None
        B = None
        for feature in range(self.NUM_FEATURES):
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
        return j, B

    def growTree(self, S, depth):
        if self.checkIfLeaf(S) or depth >= self.max_depth or len(S) <= 25:
            return LeafNode(self.getMajority(S))
        else:
            j, B = self.segmenter(S)
            left = []
            right = []
            feature_vector = getFeature(S, j)
            if j not in categorical:
                for i in range(len(feature_vector)):
                    if feature_vector[i] < B:
                        left.append(S[i])
                    else:
                        right.append(S[i])
            else:
                for i in range(len(feature_vector)):
                    if feature_vector[i] == B:
                        left.append(S[i])
                    else:
                        right.append(S[i])
            left = np.array(left)
            right = np.array(right)
            return InNode(j, B, self.growTree(left, depth + 1), self.growTree(right, depth + 1))

    def train(self, S):
        self.LABEL = len(S[0]) - 1
        self.NUM_FEATURES = self.LABEL - 1
        self.NUM_SAMPLES = len(S)
        self.root = self.growTree(S, 0)

    def predict(self, test):
        predictions = []
        for sample in test:
            curr = self.root
            count = 0
            print("START")
            while not curr.isLeaf:
                j = curr.getFeature()
                B = curr.getSplit()
                count+=1
                if j not in categorical:
                    if sample[j] < B:
                        curr = curr.left
                        print(j, B, "left")
                    else:
                        curr = curr.right
                        print(j, B, "right")
                else:
                    if sample[j] == B:
                        curr = curr.left
                        print(j, B, "left")
                    else:
                        curr = curr.right
                        print(j, B, "right")
            print(curr.getLabel())
            predictions.append(int(curr.getLabel()))
        return predictions
