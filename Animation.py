import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Input Parameters via User Input
def get_float_input(prompt):
    """
    Get Float
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Please Enter A NUMBER")

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

print("\nFconstant:")
Fx = get_float_input("  Fx: ")
Fy = get_float_input("  Fy: ")
F = np.array([Fx, Fy])

t_max = get_float_input("\ntime: ")


dt = 0.02        

#Calculations

#Calculate acceleration
a = F / m
print(f"Mass: {m} kg")
print(f"Initial Position: {r0} m")
print(f"Initial Velocity: {v0} m/s")
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

#Animation
fig, ax = plt.subplots(figsize=(8, 6))


line, = ax.plot([], [], 'b-', lw=2, label='Trajectory') # 
line2, = ax.plot([], [], 'g--', lw=2, label='Intended Trajectory')
point, = ax.plot([], [], 'ro', markersize=8, label=f'Object (m={m}kg)') # 
time_template = 'Time = %.2fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes) # 


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
    point.set_data([], [])
    time_text.set_text('')
    return line, line2, point, time_text


def update(frame):
    
    current_t = t_values[frame]
    line.set_data(x_traj[:frame+1], y_traj[:frame+1])
    line2.set_data(x_traj_no_F[:frame+1], y_traj_no_F[:frame+1])
    point.set_data([x_traj[frame]], [y_traj[frame]])
    time_text.set_text(time_template % current_t)

    return line, line2, point, time_text


ani = animation.FuncAnimation(fig, update, frames=num_frames,
                            init_func=init, blit=True, interval=dt*1000, repeat=False)


plt.show()


