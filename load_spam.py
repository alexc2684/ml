import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

train_data = sio.loadmat("dist/spam_data.mat")['training_data']
train_labels = sio.loadmat("dist/spam_data.mat")['training_labels']
test_data = sio.loadmat("dist/spam_data.mat")['test_data']
NUM_FEATURES = len(train_data[0])
NUM_CLASSES = 2
NUM_SAMPLES = len(train_data)

tdata = np.array(train_data)
tlabels = np.array(train_labels)

data = np.append(tdata, np.transpose(tlabels), 1)
np.random.shuffle(data)

validation = data[int(NUM_SAMPLES*.8):]
train = data[:int(NUM_SAMPLES*.8)]
train_features = np.delete(train, [NUM_FEATURES], axis=1)
train_labels = train[:, NUM_FEATURES]


#sort data by class
sorted_data = data[data[:,NUM_FEATURES].argsort()]
features = np.delete(sorted_data, [NUM_FEATURES], axis=1)
labels = sorted_data[:, NUM_FEATURES]
features = normalize(np.array(features, dtype="float64"),norm='l2')

classes = [] #list of classes
means = [] #mean vectors of each class
cov_matrices = []
class_samples = []
priors = np.zeros(NUM_CLASSES)

#group samples into their respective classes
cl = 0
for i in range(len(sorted_data)):
    priors[cl] += 1
    if labels[i] != cl:
        cl += 1
        classes.append(np.array(class_samples))
        class_samples = []
        class_samples.append(features[i])
    else:
        class_samples.append(features[i])
classes.append(class_samples)

priors /= NUM_SAMPLES
print(priors)
print(classes)

for c in classes:
    m = np.array(c).mean(axis=0)
    m = np.array(m).reshape((1,NUM_FEATURES))
    means.append(m)

for i in range(NUM_CLASSES):
    cov_matrix = np.zeros((NUM_FEATURES,NUM_FEATURES))
    mean_vector = means[i]

    for sample in classes[i]:
        sample = sample.reshape(1,NUM_FEATURES) #transform into column vector
        cov_matrix += np.outer((sample - mean_vector), (sample - mean_vector).T)
    cov_matrix /= len(classes[i])
    cov_matrices.append(cov_matrix)

sigma = np.zeros((NUM_FEATURES,NUM_FEATURES))
for cov in cov_matrices:
    sigma += cov
sigma = sigma/NUM_CLASSES
eig_vals, eig_vectors = np.linalg.eig(sigma)
eig_vals = eig_vals[np.nonzero(eig_vals)]
best = 99999999
for i in eig_vals:
    if i > 0 and i < best:
        best = i
min_eig = best
sigma += np.multiply(np.identity(NUM_FEATURES),min_eig)
sigma_inv = np.linalg.inv(sigma)

np.save('spam_means.npy', means)
np.save('spam_sigma.npy', sigma)
np.save('spam_sigma_inv.npy', sigma_inv)
np.save('spam_cov_matrices.npy', cov_matrices)
np.save('spam_priors.npy', priors)
