train_data = sio.loadmat("hw6_data_dist/letters_data.mat")['train_x']
train_labels = sio.loadmat("hw6_data_dist/letters_data.mat")['train_y']
test = sio.loadmat("hw6_data_dist/letters_data.mat")['test_x']

NUM_FEATURES = len(train_data[0]) - 1
NUM_CLASSES = 26
NUM_SAMPLES = len(train_data)
NUM_TEST = len(test)
NUM_VAL = 98840

train_data = np.array(train_data)
# train_data = np.append(train_data, np.reshape(np.ones(NUM_SAMPLES), (NUM_SAMPLES, 1)), 1)
train_data = np.append(train_data, train_labels, 1)
np.random.shuffle(train_data)

train_labels = train_data[:,NUM_FEATURES + 1]
train_labels = train_labels.reshape((len(train_labels),1))

train_data = train_data[:,:NUM_FEATURES + 1]
enc = OneHotEncoder()
train_labels = enc.fit_transform(train_labels).toarray()


# test = np.append(test, np.reshape(np.ones(NUM_TEST), (NUM_TEST, 1)), 1)
test = np.array(test)

ss = StandardScaler().fit(train_data)
train_data = ss.transform(train_data)
test = ss.transform(test)

validation = train_data[NUM_VAL:]
val_labels = train_labels[NUM_VAL:]

train_samples = train_data[:NUM_VAL]
train_labels = train_labels[:NUM_VAL]

def log_loss(Z, Y):
    return -sum([Y[i]*np.log(Z[i]) + (1-Y[i]*np.log(1-Z[i])) for i in range(len(Z))])

def sigmoid(x):
    return expit(x)

def nabla_w(z, y, h):
    grad = np.outer((z-y), h.T)
    return grad

def nabla_h(W, z, y):
    return W.T@(z-y)

def nabla_v(nabla_h, h, x):
    h = h[:200]
    h = np.reshape(h, (200))
    dh = np.diag(np.ones(200)) - np.diag(h**2)
    nh = nabla_h[:200]
    nh = np.reshape(nh, (200,1))
    return dh@nh@x

def train_net(images, labels, epochs, ev, ew):
    #V = 200 x 785 weight matrix, last column is biases (multiplied by 1)
    V = np.ndarray(shape=(200,785))
    #W = 26 x 201 weight matrix
    W = np.ndarray(shape=(26,201))

    e_v = ev
    e_w = ew
    l = 0.1
    count = 0

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i,j] = np.random.normal(0,.01)


    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] = np.random.normal(0,.01)

    losses = []
    for epoch in range(epochs):
        arr = np.random.choice(len(images), len(images), replace=False)
        for i in arr:
            index = i
            sample = images[index]
            sample = np.append(sample, 1)
            sample = np.reshape(sample, (len(sample),1))
            label = labels[index]
            label = label.reshape((len(label),1))
            h = V@sample
            h = np.tanh(h)
            h = np.append(h, 1)
            z = W@h
            z = expit(z)
            z = z.reshape((len(z),1))

            losses.append(log_loss(z, label))
            if i % 5000 == 0:
                loss.append(np.mean(losses))
                losses.clear()

            grad_v = nabla_v(nabla_h(W, z, label), h, sample.T)
            grad_w = nabla_w(z, label, h)

            V = V - e_v*grad_v
            W = W - e_w*grad_w

        e_v *= .95
        e_w *= .95
    return (V, W)

def predict(sample, V, W):
    sample = np.append(sample, 1)
    sample = np.reshape(sample, (len(sample),1))
    h = V@sample
    h = np.tanh(h)
    h = np.append(h, 1)
    h = h.reshape((len(h)))
    z = W@h
    z = expit(z)
    index = np.argmax(z)
    return index

def test_accuracy(samples, labels, V, W):
    count = 0
    for i in range(len(samples)):
        index = i
        sample = samples[index]
        sample = np.reshape(sample, (len(sample),1))
        label = np.argmax(labels[index])
        prediction = predict(sample, V, W)
        if prediction == label:
            count += 1
    return count/len(samples)

V, W = train_net(train_samples, train_labels, 10, .01, .001)

print(test_accuracy(validation, val_labels, V, W))
