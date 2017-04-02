class Node:
    isLeaf = False

class InNode(Node):
    def __init__(self, feature, split, left, right):
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right

    def getFeature(self):
        return self.feature

    def getSplit(self):
        return self.split

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

class LeafNode(Node):
    def __init__(self, label):
        self.label = label
        self.isLeaf = True

    def getLabel(self):
        return self.label
