import numpy as np
import pandas as pd
import scipy.optimize

offset = 1
np.random.seed(1)
layers = 128

SiLU = lambda x: x / (1 + np.exp(-x))
d_SiLU = lambda x: x * (1 + np.exp(-x))**-2 * np.exp(-x)

x = np.linspace(0, 10, 10)
y = np.linspace(0, 10, 10)
t = np.linspace(0, 10, 10)
W1 = np.random.randn(layers, 3) * 0.01
b1 = np.zeros((layers, 1))
W2 = np.random.randn(3, layers) * 0.01
b2 = np.zeros((3, 1))

p_flat = np.concatenate([
    W1.ravel(), 
    b1.ravel(), 
    W2.ravel(), 
    b2.ravel()
])

#makes grid of x, y, t easily usable
def generalize_data(x, y, t):
    X, Y, T = np.meshgrid(x, y, t)
    X = X.flatten()
    Y = Y.flatten()
    T = T.flatten()
    S = np.stack([X, Y, T])
    return S

#returns physical residual error
def R_p(raw_out, h, u, v, da1_dz1, W1, W2):
     # Derivative of our Softplus function wrapper on h: dh/draw_out
    dh_draw = 1.0 / (1.0 + np.exp(-raw_out[0:1, :]))

    # --- ANALYTICAL CALCULUS VIA NUMPY MATRICES ---

    # Extract weights explicitly for clarity
    # W2[0, :] is the row of 128 weights mapped to output h
    w2_h = W2[0:1, :]  # Shape: (1, 128)

    # Compute dh/dx, dh/dy, dh/dt across all 200,000 points continuously
    # We map the chain rule backward from the output layer through Layer 1 weights
    dh_dx = dh_draw * (w2_h @ (da1_dz1 * W1[:, 0:1]))  # W1[:, 0] is weights for x
    dh_dy = dh_draw * (w2_h @ (da1_dz1 * W1[:, 1:2]))  # W1[:, 1] is weights for y
    dh_dt = dh_draw * (w2_h @ (da1_dz1 * W1[:, 2:3]))  # W1[:, 2] is weights for t

    # Compute du/dx and dv/dy (No softplus wrapper on velocities, so no dh_draw term)
    w2_u = W2[1:2, :]  # Shape: (1, 128)
    w2_v = W2[2:3, :]  # Shape: (1, 128)

    du_dx = w2_u @ (da1_dz1 * W1[:, 0:1])
    dv_dy = w2_v @ (da1_dz1 * W1[:, 1:2])

    physics_residual = dh_dt + (u * dh_dx + h * du_dx) + (v * dh_dy + h * dv_dy)
    return physics_residual

#runs model, # W is 128 x 3
def model(S, p0):
    W1 = p0[0 : 3*layers].reshape(layers, 3)
    b1 = p0[3*layers : 4*layers].reshape(layers, 1)
    W2 = p0[4*layers : 7*layers].reshape(3, layers)
    b2 = p0[7*layers :].reshape(3, 1)
    z1 = (W1 @ S) + b1              # Shape: (128, 200000)
    a1 = SiLU(z1)                      # Shape: (128, 200000)
    da1_dz1 = d_SiLU(z1)       
    raw_out = (W2 @ a1) + b2     #row 0 = h, row 1 = u, row 2 = v
    return raw_out, da1_dz1, W1, W2

#returns sum of squared error with penalization for going against PDEs
def SSE(p, S):
    raw_out, da1_dz1, W1, W2 = model(S, p)
    h = np.log(1.0 + np.exp(raw_out[0:1, :])) # Shape: (1, 200000)
    u = raw_out[1:2, :]                       # Shape: (1, 200000)
    v = raw_out[2:3, :]                       # Shape: (1, 200000)

    #Calculate MSE
    L_data = 0       #set to 0 if no historical data exists
    L_physical = np.sum(R_p(raw_out, h, u, v, da1_dz1, W1, W2)**2)
    return L_data + offset * L_physical

S = generalize_data(x, y, t)
result = scipy.optimize.minimize(SSE, p_flat, args = (S), method = 'L-BFGS-B').x
print(result)
