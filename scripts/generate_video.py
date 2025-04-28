import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Load logged simulation data
data = np.load('results/robot_log.npz')
qpos = data['qpos']  # joint positions
ctrl = data['ctrl']  # torques
timesteps = qpos.shape[0]
num_joints = qpos.shape[1]

# Set up the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot joint positions
lines_pos = [axs[0].plot([], [], label=f'Joint {i}')[0] for i in range(num_joints)]
axs[0].set_xlim(0, timesteps)
axs[0].set_ylim(np.min(qpos) * 1.2, np.max(qpos) * 1.2)
axs[0].set_title('Joint Positions Over Time')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Position (rad)')
axs[0].legend()
axs[0].grid()

# Plot control torques
lines_ctrl = [axs[1].plot([], [], label=f'Joint {i}')[0] for i in range(ctrl.shape[1])]
axs[1].set_xlim(0, timesteps)
axs[1].set_ylim(np.min(ctrl) * 1.2, np.max(ctrl) * 1.2)
axs[1].set_title('Control Inputs (Torques) Over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Torque (Nm)')
axs[1].legend()
axs[1].grid()

# Initialize empty data
def init():
    for line in lines_pos + lines_ctrl:
        line.set_data([], [])
    return lines_pos + lines_ctrl

# Animate frame-by-frame
def animate(i):
    for j, line in enumerate(lines_pos):
        line.set_data(np.arange(i), qpos[:i, j])
    for j, line in enumerate(lines_ctrl):
        line.set_data(np.arange(i), ctrl[:i, j])
    return lines_pos + lines_ctrl

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=timesteps, interval=30, blit=True)

# Save animation to MP4
os.makedirs('results', exist_ok=True)
ani.save('results/simulation_plot_video.mp4', writer='ffmpeg', fps=30)

print("\u2705 Video saved at results/simulation_plot_video.mp4")
