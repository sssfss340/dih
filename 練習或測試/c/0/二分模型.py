import numpy as np
from sklearn.datasets import load_breast_cancer

layer=int(input("layer"))
learn_rate=0.1
train=int(input("train"))
data = load_breast_cancer()
feature_ori=data["data"][:501]
ans=data["target"][:501].reshape(-1, 1)
step=int(np.ceil(feature_ori.shape[1]/layer))
w=[]
b=[]
for node in range(feature_ori.shape[1],1,-step):
    w.append(np.random.randn(node, max(node-step, 1)) * np.sqrt(2.0 / node))
    b.append(np.zeros((1, max(node-step, 1))))
for arr in w:
    print(arr.shape)
for t in range(train): 
    feature=np.array(feature_ori,copy=True)
    a=[]
    for l in range(layer-1):
        z=feature@w[l]+b[l]
        relu=np.maximum(0,z)
        feature=relu
        a.append(relu)
    z=feature@w[layer-1]+b[layer-1]
    clipped_z = np.clip(z, -500, 500)
    sigmoid=1/(1+np.exp(-clipped_z))
    y_hat=sigmoid

    loss=-ans*np.log(y_hat+ 1e-15)-(1-ans)*np.log(1-y_hat+ 1e-15)
    print(np.sum(loss)/feature_ori.shape[0])

    dz = sigmoid - ans
    grad_w = [None] * layer
    grad_b = [None] * layer
    for back in range(layer - 1, -1, -1):
        prev_a = feature_ori if back == 0 else a[back - 1]
        grad_w[back] = prev_a.T @ dz
        grad_b[back] = np.sum(dz, axis=0, keepdims=True)
        if back > 0:
            da = dz @ w[back].T
            dz = da * (a[back - 1] > 0)
    for l in range(layer):
        w[l] = w[l] - learn_rate/feature_ori.shape[0] * grad_w[l]
        b[l] = b[l] - learn_rate/feature_ori.shape[0] * grad_b[l]

fea_test=data["data"][501:]
feature=np.array(fea_test,copy=True)
a=[]
for l in range(layer-1):
    z=feature@w[l]+b[l]
    relu=np.maximum(0,z)
    feature=relu
    a.append(relu)
z=feature@w[layer-1]+b[layer-1]
clipped_z = np.clip(z, -500, 500)
sigmoid=1/(1+np.exp(-clipped_z))
y_hat=sigmoid
ans=data["target"][501:].reshape(-1, 1)
loss=-ans*np.log(y_hat)-(1-ans)*np.log(1-y_hat)
print(np.sum(loss)/fea_test.shape[0])