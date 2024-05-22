import numpy as np
import matplotlib.pyplot as plt

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cahe = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # Affine 
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # Slicing
        f = A[:,:,H]
        g = A[:,H:2*H] 
        i = A[:, 2*H : 3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f*c_prev+g*i
        h_next = np.tanh(c_next)*o

        self.cache = (x, h_prev, c_prev, f, g, i, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params

        x,h_prev,c_prev,f,g,i,o,c_next = self.cache
        tanh_c_next = np.tanh(c_next)

        ds = dh_next*o*(1-tanh_c_cext**2)+dc_next
        dc_prev = ds*f

        
    
        

# class Affine:
#     def __init__(self, W, b):
#         self.W = W
#         self.b = b
#         self.x = None
#         self.dW = None
#         self.db = None

#     def forward(self, x):
#         self.x = x
#         out = np.matmul(x, self.W)+self.b
#         return out
    
#     def backward(self, d_out):
#         dx = np.matmul(d_out, self.W.T)
#         self.dW = np.matmul(self.xT, d_out)
#         self.db = np.sum(d_out, axis = 0)
#         return dx
