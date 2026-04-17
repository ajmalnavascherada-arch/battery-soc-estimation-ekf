import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Fake Battery Data
# -----------------------------
def generate_data(N=500):
    dt = 1
    time = np.arange(N)

    current = 2 * np.sin(0.02 * time)  # battery usage

    soc_true = np.zeros(N)
    soc_true[0] = 0.9  # start at 90%

    Q = 2.3 * 3600  # capacity

    for k in range(1, N):
        soc_true[k] = soc_true[k-1] - (current[k] * dt) / Q

    voltage = (
        3 + 0.8*soc_true - 0.1*soc_true**2
        - 0.01*current
        + 0.01*np.random.randn(N)
    )

    return current, voltage, soc_true, time


# -----------------------------
# Step 2: EKF (Simple Version)
# -----------------------------
def ekf(current, voltage):
    N = len(current)

    Q_batt = 2.3 * 3600
    R0 = 0.01
    R1 = 0.02
    C1 = 2000
    dt = 1

    x = np.array([0.9, 0.0])  # [SoC, V1]
    P = np.eye(2) * 0.01

    Qk = np.diag([1e-6, 1e-5])
    Rk = 0.01

    soc_est = np.zeros(N)

    for k in range(N):
        I = current[k]

        # -------- Prediction --------
        a = np.exp(-dt/(R1*C1))

        x_pred = np.array([
            x[0] - (I*dt)/Q_batt,
            a*x[1] + R1*(1-a)*I
        ])

        P = P + Qk

        # -------- Measurement --------
        V_pred = (
            3 + 0.8*x_pred[0] - 0.1*x_pred[0]**2
            - I*R0 - x_pred[1]
        )

        error = voltage[k] - V_pred

        # -------- Update --------
        K = 0.1  # simplified gain
        x = x_pred + K * error

        soc_est[k] = x[0]

    return soc_est


# -----------------------------
# Step 3: Run Everything
# -----------------------------
current, voltage, soc_true, time = generate_data()
soc_est = ekf(current, voltage)

# -----------------------------
# Step 4: Plot
# -----------------------------
plt.figure()
plt.plot(time, soc_true, label="True SoC")
plt.plot(time, soc_est, '--', label="Estimated SoC")
plt.xlabel("Time")
plt.ylabel("SoC")
plt.title("EKF Battery SoC Estimation")
plt.legend()
plt.grid()

plt.show()
rmse = np.sqrt(np.mean((soc_true - soc_est)**2))
print(f"RMSE: {rmse:.5f}")
