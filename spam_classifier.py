import math
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize


train_data = sio.loadmat("dist/spam_data.mat")['training_data']
train_labels = sio.loadmat("dist/spam_data.mat")['training_labels']
test_data = sio.loadmat("dist/spam_data.mat")['test_data']
NUM_FEATURES = len(train_data[0])
NUM_CLASSES = 2
NUM_SAMPLES = len(train_data)
PARTITION = int(NUM_SAMPLES*.8)

tdata = np.array(train_data)
tlabels = np.array(train_labels)
test = np.array(test_data)

data = np.append(tdata, np.transpose(tlabels), 1)
np.random.shuffle(data)

train = data[:PARTITION]
train_features = np.delete(train, [NUM_FEATURES], axis=1)
train_features = normalize(np.array(train_features, dtype="float64"),norm='l2')
train_labels = train[:, NUM_FEATURES]
train_labels = train_labels.astype(int)

validation = data[PARTITION:]
validation_features = np.delete(validation, [NUM_FEATURES], axis=1)
validation_features = normalize(np.array(validation_features, dtype="float64"),norm='l2')
validation_labels = validation[:, NUM_FEATURES]


means = np.load("spam_means.npy")
cov_matrices = np.load("spam_cov_matrices.npy")
sigma = np.load("spam_sigma.npy")
sigma_inv = np.load("spam_sigma_inv.npy")
priors = np.load("spam_priors.npy")
determinant = np.linalg.det(sigma)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_lda(sample):
    best = -99999999
    c = None
    for i in range(NUM_CLASSES):
        f = np.dot(np.dot(train_means[i], sigma_inv), sample)
        s = .5*np.dot(np.dot(train_means[i], sigma_inv), train_means[i].T)
        t = np.log(priors[i])
        score = f + s + t
        score = score[0][0]
        if score > best:
            best = score
            c = i
    return c

def predict_qda(sample):
    best = -99999999
    c = None
    sample = sample.reshape((1,NUM_FEATURES))
    for i in range(NUM_CLASSES):
        diff = (sample - train_means[i])
        f = -.5*np.dot(np.dot(diff, sigma_inv), diff.T)
        s = -.5*math.pow(np.log(2*math.pi), NUM_FEATURES)*determinant
        t = np.log(priors[i])
        score = f[0][0] + s + t
        # print(str(i) + ":" + str(score))
        if score > best:
            best = score
            c = i
    return c

vals = [PARTITION] #[100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
results = []
for NUM_TRAIN in vals:
    train_samples = []
    for i in range(NUM_CLASSES):
        train_samples.append([])
    train_means = []
    for i in range(NUM_TRAIN):
        X = train_features[i]
        y = train_labels[i]
        train_samples[y].append(X)

    #calc means
    for c in train_samples:
        m = np.array(c).mean(axis=0)
        m = np.array(m).reshape((1,NUM_FEATURES))
        train_means.append(m)

    #validation
#     count = 0
#     for i in range(len(validation_features)):
#         sample = np.array(validation_features[i])
#         prediction = predict_lda(sample) #switch this line when using qda
#         if prediction == validation_labels[i]:
#             count += 1
#     results.append((1 - (count/len(validation_features)))*100)
#     print(str(NUM_TRAIN) + ":" + str((1 - (count/len(validation_features)))*100))
# print(results)

test_predictions = []
for sample in test:
    prediction = predict_qda(sample)
    test_predictions.append(prediction)

predictions = "Id,Category\n"
index = 0
for guess in test_predictions:
    predictions += str(index) + "," + str(guess) + "\n"
    index += 1
file = open("spam_predictions.csv", "w")
file.write(predictions)

#display the covariance matrix
# plt.imshow(cov_matrices[0], cmap='hot', interpolation='nearest')
# plt.show()
