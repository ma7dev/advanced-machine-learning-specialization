import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

with open('train.npy', 'rb') as fin:
    X = np.load(fin)

with open('target.npy', 'rb') as fin:
    y = np.load(fin)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
plt.show()

def expand(X):
    X_expanded = np.zeros((X.shape[0], 6))

    for feature in X:
        X_expanded[i][0] = feature[0]
        X_expanded[i][1] = feature[1]
        X_expanded[i][2] = np.square(feature[0])
        X_expanded[i][3] = np.square(feature[1])
        X_expanded[i][4] = np.multiply(feature[0], feature[1])
        X_expanded[i][5] = 1
    return X_expanded

X_expanded = expand(X)

# simple test on random numbers

dummy_X = np.array([
        [0,0],
        [1,0],
        [2.61,-1.28],
        [-0.59,2.1]
    ])

# call your expand function
dummy_expanded = expand(dummy_X)

# what it should have returned:   x0       x1       x0^2     x1^2     x0*x1    1
dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],
                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],
                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],
                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])

#tests
assert isinstance(dummy_expanded,np.ndarray), "please make sure you return numpy array"
assert dummy_expanded.shape == dummy_expanded_ans.shape, "please make sure your shape is correct"
assert np.allclose(dummy_expanded,dummy_expanded_ans,1e-3), "Something's out of order with features"

print("Seems legit!")

def probability(X, w):
    segma = (1)/(1+np.exp(-np.inner(w,X)))
    return segma

dummy_weights = np.linspace(-1, 1, 6)
ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]

def compute_loss(X, y, w):
    L = 0
    for i, x in enumerate(X):
        L += -((y[i]*np.log(probability(x, w)))+((1-y[i])*np.log(1-probability(x, w))))
    L /= X.shape[0]
    return L

# use output of this cell to fill answer field
ans_part2 = compute_loss(X_expanded, y, dummy_weights)

def compute_grad(X, y, w):
    d_L = 0
    for i, x in enumerate(X):
        d_L += (probability(x, w)-y[i])*x  # how? By hand?
    d_L /= X.shape[0]
    return d_L


# use output of this cell to fill answer field
ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))

from IPython import display

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()

visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])

# please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # Keep in mind that compute_grad already does averaging over batch for you!
    w[i] = w[i-1] - eta * compute_grad(X_expanded, y, w)

visualize(X, y, w, loss)
plt.clf()

# use output of this cell to fill answer field

ans_part4 = compute_loss(X_expanded, y, w)


# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    nu = alpha*nu + eta * compute_grad(X_expanded, y, w)
    w = w - nu

visualize(X, y, w, loss)
plt.clf()

# use output of this cell to fill answer field
ans_part5 = compute_loss(X_expanded, y, w)

# please use np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1.])

eta = 0.1 # learning rate
alpha = 0.9 # moving average of gradient norm squared
g2 = None # we start with None so that you can update this value correctly on the first iteration
eps = 1e-8

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    g2 = alpha*g2 + (1-alpha)*np.square(compute_grad(X_expanded, y, w))
    w = w - (eta/(np.sqrt(g2+eps)))*compute_grad(X_expanded, y, w)

visualize(X, y, w, loss)
plt.clf()

# use output of this cell to fill answer field
ans_part6 = compute_loss(X_expanded, y, w)
