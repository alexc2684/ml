import numpy as np
import scipy.io as sio
import scipy.special as spe
import math
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
train_data = sio.loadmat("data.mat")["X"]
train_labels = sio.loadmat("data.mat")["y"]
test_data = sio.loadmat("data.mat")["X_test"]

train_data = normalize(np.array(train_data, dtype="float64"),norm='l2')
train_labels = np.array(train_labels)
test_data = normalize(np.array(test_data, dtype="float64"),norm='l2')

NUM_FEATURES = len(train_data[0])
NUM_SAMPLES = len(train_data)
LAMBDA = .00001
EPSILON = .7
NUM_VAL = NUM_SAMPLES - NUM_SAMPLES//5
NUM_TRAIN = NUM_SAMPLES - NUM_VAL

val_data = train_data[NUM_VAL:]
val_labels = train_labels[NUM_VAL:]
train_data = train_data[:NUM_VAL]
train_labels = train_labels[:NUM_VAL]
weights = np.zeros((NUM_FEATURES))
weights = weights.reshape((12,1))

def sigmoid(x):
    return spe.expit(x)

def arr_sigmoid(x):
    s = [sigmoid(i) for i in x]
    return np.array(s)

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def arr_sigmoid_prime(x): #returns matrix w s' on diagonals
    omega = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        omega[i,i] = sigmoid_prime(x[i])
    return omega

def batch_GD(w, X, y):
    cst = (X.T)@(np.subtract(y, sigmoid(np.dot(X,w))))
    cst = np.divide(cst, len(X))
    l2 = np.multiply(2*LAMBDA, w)
    return np.multiply(EPSILON, cst - l2)

def stochastic_GD(w, X, y):
    num = np.random.choice(len(X))
    xi = np.array(X[num])
    xi = xi.reshape(1,12)
    cst = (np.subtract(y[num], sigmoid(xi@w)))@xi
    cst = np.divide(cst, len(X))
    cst = cst.reshape((12,1))
    l2 = np.multiply(2*LAMBDA, w)
    return np.multiply(EPSILON, cst - l2)


def l2_cost(w, X, y):
    total = 0
    for i in range(NUM_TRAIN):
        x = np.array(X[i])
        dot = (np.dot(x,w))[0]
        z = sigmoid(dot)
        val = y[i]*np.log(z) + (1-y[i])*np.log(1-z)
        total += val
    l2 = LAMBDA*math.sqrt(sum([a**2 for a in w]))
    total += l2
    return -(total/NUM_TRAIN)

its = [i for i in range(500)]
# bc = []

# batch gradient descent
#TRAIN
for i in range(500):
    weights = np.add(weights, batch_GD(weights, train_data, train_labels))
    cost = l2_cost(weights, train_data, train_labels)
    print(i)

#cross validation
def cross_validation():
    best = 100000
    lam = None
    ls = [i*.05 for i in range(20)]
    for l in ls:
        EPSILON = l
        for i in range(100):
            weights = np.add(weights, batch_GD(weights, train_data, train_labels))
            cost = l2_cost(weights, train_data, train_labels)
        if cost < best:
            best = cost
            lam = EPSILON
    return lam

# stochastic gradient descent
for i in range(500):
    weights = np.add(weights, stochastic_GD(weights, train_data, train_labels))
    cost = l2_cost(weights, train_data, train_labels)
    sc.append(cost)

for i in range(500):
    weights = np.add(weights, stochastic_GD(weights, train_data, train_labels))
    cost = l2_cost(weights, train_data, train_labels)
    bce.append(cost)
    EPSILON = 1/(i+1)

# validation
num_val = len(val_data)
count = 0
for i in range(num_val):
    guess = sigmoid(val_data[i]@weights)[0]
    guess = np.rint(guess)
    guess = int(guess)
    if guess == val_labels[i]:
        count += 1
print("Score:", count/num_val)

test_predictions = []
for sample in test_data:
    guess = sigmoid(sample@weights)[0]
    guess = np.rint(guess)
    guess = int(guess)
    test_predictions.append(guess)

predictions = "Id,Category\n"
index = 0
for guess in test_predictions:
    predictions += str(index) + "," + str(guess) + "\n"
    index += 1
file = open("predictions.csv", "w")
file.write(predictions)
