import numpy as np
import matplotlib.pyplot as plt

def plot_results(filename="robot_log.npz"):
    data = np.load(filename)
    time = data['time'] - data['time'][0]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    for i in range(7):
        plt.plot(time, [q[i] for q in data['qpos']])
    plt.title('Joint Positions')
    
    plt.subplot(3, 1, 2)
    for i in range(7):
        plt.plot(time, [c[i] for c in data['ctrl']])
    plt.title('Control Signals')
    
    plt.subplot(3, 1, 3)
    for i in range(7):
        plt.plot(time, [v[i] for v in data['qvel']])
    plt.title('Joint Velocities')
    
    plt.tight_layout()
    plt.show()