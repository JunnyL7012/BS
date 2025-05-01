import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

st.title("Black-Scholes vs Delta Hedging Simulation")

S = st.number_input("Initial Stock Price (S)", value=100.0)
X = st.number_input("Strike Price (X)", value=100.0)
T = st.number_input("Time to Maturity (T, in years)", value=1.0)
r = st.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
Nrep = st.number_input("Number of Simulations", value=1000, step=100)
Nstep = st.number_input("Steps per Simulation", value=100, step=10)

def bs_price(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

def delta(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

dt = T / Nstep
paths = np.zeros((Nrep, Nstep + 1))
paths[:, 0] = S

for i in range(Nrep):
    for j in range(1, Nstep + 1):
        z = np.random.normal()
        paths[i, j] = paths[i, j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

costs = np.zeros(Nrep)

for i in range(Nrep):
    cf = np.zeros(Nstep + 1)
    position = 0

    for j in range(Nstep):
        t = j * dt
        T_left = T - t
        dlt = delta(paths[i, j], X, T_left, r, sigma)
        cf[j] = (position - dlt) * paths[i, j]
        position = dlt

    if paths[i, -1] > X:
        cf[-1] = X - (1 - position) * paths[i, -1]
    else:
        cf[-1] = position * paths[i, -1]

    discount_factors = np.exp(-r * dt * np.arange(Nstep + 1))
    costs[i] = -np.sum(cf * discount_factors)

bs = bs_price(S, X, T, r, sigma)
dh = np.mean(costs)
diff = dh - bs

st.subheader("Result")
st.write(f"Black-Scholes Price: {bs:.4f}")
st.write(f"Delta Hedging Price: {dh:.4f}")
st.write(f"Difference: {diff:.4f}")

st.subheader("Sample Stock Price Paths")
fig, ax = plt.subplots()
for i in range(min(10, Nrep)):
    ax.plot(paths[i])
ax.set_title("Stock Price Simulation Paths")
ax.set_xlabel("Time Step")
ax.set_ylabel("Price")
st.pyplot(fig)
