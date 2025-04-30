import numpy as np

# linear classifier prediction function
def predict(x, W, b):
    return np.dot(W, x) + b

x = np.array([193, 72, 233, 255]) # input instance

# weight (parameter) matrix
W = np.matrix([[-0.84, 0.46, 0.35, -0.32],
              [0.08, 0.66, 0.91, 0.81],
              [-0.20, 0.63, 0.79, -0.23]])

b = np.array([-5.36, -0.27, 0.46]) # bias

print(predict(x, W, b))