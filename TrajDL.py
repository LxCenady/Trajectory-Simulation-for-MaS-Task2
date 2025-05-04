import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import time

# Input Parameters via User Input
def get_float_input(prompt):
    """
    Get Float
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print(" Please Enter A NUMBER")

print("Input：")

m = get_float_input("Mass: ")

print("\nCoordinate:")
x0 = get_float_input("  Xinit: ")
y0 = get_float_input("  Yinit: ")
r0 = np.array([x0, y0])

print("\nVinit:")
vx0 = get_float_input("  Vx: ")
vy0 = get_float_input("  Vy: ")
v0 = np.array([vx0, vy0])
v0_magnitude = np.linalg.norm(v0)  

print("\nFconstant:")
Fx = get_float_input("  Fx: ")
Fy = get_float_input("  Fy: ")
F = np.array([Fx, Fy])

t_max = get_float_input("\ntime: ")

dt = 0.02        

# Calculations
# Calculate acceleration
a = F / m
print(f"Mass: {m} kg")
print(f"Initial Position: {r0} m")
print(f"Initial Velocity: {v0} m/s (magnitude: {v0_magnitude:.2f} m/s)")
print(f"Force: {F} N")
print(f"Acceleration: {a} m/s^2")

t_values = np.arange(0, t_max + dt, dt)
num_frames = len(t_values)
t_col = t_values[:, np.newaxis]
positions = r0 + v0 * t_col + 0.5 * a * t_col**2
positions_no_F = r0 + v0 * t_col
x_traj_no_F, y_traj_no_F = positions_no_F[:, 0], positions_no_F[:, 1]
x_traj = positions[:, 0]
y_traj = positions[:, 1]

# Animation
fig, ax = plt.subplots(figsize=(10, 8))

line, = ax.plot([], [], 'b-', lw=2, label='Trajectory') 
line2, = ax.plot([], [], 'g--', lw=2, label='Intended Trajectory (No Force)')
optimized_line, = ax.plot([], [], 'm-', lw=2, label='Optimized Trajectory')
point, = ax.plot([], [], 'ro', markersize=8, label=f'Object (m={m}kg)')
target_point, = ax.plot([], [], 'go', markersize=10, label='Target')
time_template = 'Time = %.2fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
iteration_template = 'Optimization Iteration: %d'
iteration_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
loss_template = 'Loss: %.4f'
loss_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)

def init():
    all_x = np.concatenate([x_traj, x_traj_no_F])
    all_y = np.concatenate([y_traj, y_traj_no_F])
    margin_x = (all_x.max() - all_x.min()) * 0.1 or 1  
    margin_y = (all_y.max() - all_y.min()) * 0.1 or 1
    
    ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
    ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Object Trajectory under Constant Force')
    ax.grid(True)
    ax.legend(loc='upper right')
    
    line.set_data([], [])
    line2.set_data([], [])
    optimized_line.set_data([], [])
    point.set_data([], [])
    target_point.set_data([], [])
    time_text.set_text('')
    iteration_text.set_text('')
    loss_text.set_text('')
    return line, line2, optimized_line, point, target_point, time_text, iteration_text, loss_text


def update(frame):
    current_t = t_values[frame]
    line.set_data(x_traj[:frame+1], y_traj[:frame+1])
    line2.set_data(x_traj_no_F[:frame+1], y_traj_no_F[:frame+1])
    point.set_data([x_traj[frame]], [y_traj[frame]])
    time_text.set_text(time_template % current_t)

    return line, line2, optimized_line, point, target_point, time_text, iteration_text, loss_text


ani = animation.FuncAnimation(fig, update, frames=num_frames,
                            init_func=init, blit=True, interval=dt*1000, repeat=False)

plt.show()


# After the initial animation is closed, prompt for target coordinate
print("\nEnter target coordinate:")
target_x = get_float_input("Target X: ")
target_y = get_float_input("Target Y: ")
target = np.array([target_x, target_y])

# Convert variables to PyTorch tensors
r0_tensor = torch.tensor(r0, dtype=torch.float32)
a_tensor = torch.tensor(a, dtype=torch.float32)
t_col_tensor = torch.tensor(t_col.squeeze(), dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32)

# Initialize theta with current angle
initial_theta = np.arctan2(v0[1], v0[0])
theta = torch.tensor(initial_theta, requires_grad=True)

# Optimization setup
optimizer = torch.optim.Adam([theta], lr=0.01)
num_iterations = 1000

# Optimization loop
for i in range(num_iterations):
    optimizer.zero_grad()
    v0_vector = v0_magnitude * torch.stack([torch.cos(theta), torch.sin(theta)])
    positions = r0_tensor + v0_vector * t_col_tensor.unsqueeze(-1) + 0.5 * a_tensor * (t_col_tensor**2).unsqueeze(-1)
    diffs = positions - target_tensor
    distances = torch.norm(diffs, dim=1)
    loss = torch.min(distances)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.4f}, Theta: {np.degrees(theta.item()):.2f}°")

# Compute optimized trajectory
optimized_theta = theta.detach().numpy()
optimized_v0 = v0_magnitude * np.array([np.cos(optimized_theta), np.sin(optimized_theta)])
positions_optimized = r0 + optimized_v0.reshape(1, -1) * t_col + 0.5 * a * t_col**2
x_opt, y_opt = positions_optimized[:, 0], positions_optimized[:, 1]

distances_opt = np.sqrt((x_opt - target_x)**2 + (y_opt - target_y)**2)
min_distance = np.min(distances_opt)
closest_frame = np.argmin(distances_opt)


if min_distance > 1.0: 
    print(f"\nError, unable to reach target. {min_distance:.2f} m）")
    import sys
    sys.exit(1)

# Update animation with optimized trajectory
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid(True) 

target_point, = ax.plot(target_x, target_y, 'y*', markersize=15, label='Target', zorder=3)

line, = ax.plot([], [], 'b-', lw=2, label='Original Trajectory')
line2, = ax.plot([], [], 'g--', lw=2, label='No Force Trajectory')
optimized_line, = ax.plot([], [], 'm-', lw=2, label='Optimized Trajectory')
point, = ax.plot([], [], 'ro', markersize=8, label=f'Object (m={m}kg)')
target_point, = ax.plot(target[0], target[1], 'go', markersize=10, label='Target')

def init():
    all_x = np.concatenate([x_traj, x_traj_no_F, x_opt])
    all_y = np.concatenate([y_traj, y_traj_no_F, y_opt])
    margin_x = (np.max(all_x) - np.min(all_x)) * 0.1 or 1
    margin_y = (np.max(all_y) - np.min(all_y)) * 0.1 or 1
    ax.set_xlim(np.min(all_x)-margin_x, np.max(all_x)+margin_x)
    ax.set_ylim(np.min(all_y)-margin_y, np.max(all_y)+margin_y)
    ax.legend(loc='upper right')
    ax.grid(True)  
    target_point.set_data([target_x], [target_y])
    return line, line2, optimized_line, point, target_point

def update(frame):
    line.set_data(x_traj[:frame+1], y_traj[:frame+1])
    line2.set_data(x_traj_no_F[:frame+1], y_traj_no_F[:frame+1])
    optimized_line.set_data(x_opt[:frame+1], y_opt[:frame+1])
    point.set_data([x_traj[frame]], [y_traj[frame]])
    return line, line2, optimized_line, point
    if frame >= closest_frame:
        ani.event_source.stop()  
        time_text.set_text(f'Reached closest point: {time_template % t_values[frame]}')
    return line, line2, optimized_line, point, target_point

ani = animation.FuncAnimation(fig, update, frames=closest_frame+1,
                            init_func=init, blit=True, interval=dt*1000, repeat=False)
    
plt.show()