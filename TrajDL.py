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
            print("请输入数字 (Please Enter A NUMBER)")

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
v0_magnitude = np.linalg.norm(v0)  # 保存初始速度大小

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

# 新增功能：目标坐标优化
print("\n原始轨迹的终点位置:", positions[-1])
print("\n现在请输入目标坐标：")
target_x = get_float_input("目标 X 坐标: ")
target_y = get_float_input("目标 Y 坐标: ")
target_position = np.array([target_x, target_y])

# 使用PyTorch进行优化
def optimize_initial_velocity(target_position, r0, a, t_max, v0_magnitude, max_iterations=200, learning_rate=0.01):
    """
    使用PyTorch优化初始速度向量，使物体能达到目标位置
    """
    # 转换为PyTorch张量
    r0_tensor = torch.tensor(r0, dtype=torch.float32)
    a_tensor = torch.tensor(a, dtype=torch.float32)
    target_tensor = torch.tensor(target_position, dtype=torch.float32)
    t_max_tensor = torch.tensor(t_max, dtype=torch.float32)
    v0_mag_tensor = torch.tensor(v0_magnitude, dtype=torch.float32)
    
    # 初始化可优化的速度方向（角度）
    initial_angle = np.arctan2(v0[1], v0[0])
    theta = torch.tensor(initial_angle, dtype=torch.float32, requires_grad=True)
    
    # 优化器
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    
    # 用于存储优化过程
    optimization_history = []
    loss_history = []
    
    # 优化循环
    for i in range(max_iterations):
        optimizer.zero_grad()
        
        # 根据角度计算速度分量（保持梯度流）
        vx = v0_mag_tensor * torch.cos(theta)
        vy = v0_mag_tensor * torch.sin(theta)
        
        # 计算终点位置：r = r0 + v*t + 0.5*a*t^2
        final_x = r0_tensor[0] + vx * t_max_tensor + 0.5 * a_tensor[0] * t_max_tensor**2
        final_y = r0_tensor[1] + vy * t_max_tensor + 0.5 * a_tensor[1] * t_max_tensor**2
        final_pos = torch.stack([final_x, final_y])
        
        # 计算损失：与目标位置的距离
        loss = torch.sqrt((final_pos[0] - target_tensor[0])**2 + (final_pos[1] - target_tensor[1])**2)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 存储当前的速度和损失
        with torch.no_grad():
            current_angle = theta.item()
            v_current = np.array([v0_magnitude * np.cos(current_angle), 
                                 v0_magnitude * np.sin(current_angle)])
            
            curr_x = r0[0] + v_current[0] * t_max + 0.5 * a[0] * t_max**2
            curr_y = r0[1] + v_current[1] * t_max + 0.5 * a[1] * t_max**2
            current_position = np.array([curr_x, curr_y])
            
            optimization_history.append(v_current)
            loss_history.append(loss.item())
            
            # 每20次迭代显示一次进度
            if i % 20 == 0 or i == max_iterations - 1:
                print(f"迭代 {i}: 损失 = {loss.item():.4f}, "
                      f"当前速度 = [{v_current[0]:.2f}, {v_current[1]:.2f}], "
                      f"预计终点 = [{current_position[0]:.2f}, {current_position[1]:.2f}]")
            
            # 如果损失足够小，提前结束
            if loss.item() < 0.01:
                print(f"提前收敛于迭代 {i}，损失 = {loss.item():.4f}")
                break
    
    # 获取最终优化的初始速度
    with torch.no_grad():
        final_angle = theta.item()
        vx_opt = v0_magnitude * np.cos(final_angle)
        vy_opt = v0_magnitude * np.sin(final_angle)
        v_opt = np.array([vx_opt, vy_opt])
        
        # 计算最终位置和距离
        final_x = r0[0] + v_opt[0] * t_max + 0.5 * a[0] * t_max**2
        final_y = r0[1] + v_opt[1] * t_max + 0.5 * a[1] * t_max**2
        final_pos = np.array([final_x, final_y])
        distance_to_target = np.linalg.norm(final_pos - target_position)
        
    return v_opt, optimization_history, loss_history, final_pos, distance_to_target

# 执行优化
print("\n开始优化初始速度...")
optimized_v0, opt_history, loss_history, final_pos, distance = optimize_initial_velocity(
    target_position, r0, a, t_max, v0_magnitude)

# 检查是否能到达目标
tolerance = 0.1  # 误差容忍度
if distance <= tolerance:
    print(f"\n优化成功！")
    print(f"优化后的初始速度: {optimized_v0} (保持了相同的速度大小: {np.linalg.norm(optimized_v0):.2f} m/s)")
    print(f"预计到达位置: {final_pos}")
    print(f"与目标的距离: {distance:.4f} m")
else:
    print(f"\n无法在保持速度大小的情况下到达目标坐标。")
    print(f"最佳近似解的初始速度: {optimized_v0}")
    print(f"最近可达位置: {final_pos}")
    print(f"与目标的距离: {distance:.4f} m")

# 使用优化后的速度计算新轨迹
optimized_positions = r0 + optimized_v0 * t_col + 0.5 * a * t_col**2
x_opt, y_opt = optimized_positions[:, 0], optimized_positions[:, 1]

# 可视化优化结果
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 绘制轨迹对比
ax1.plot(x_traj, y_traj, 'b-', label='Traj_Init')
ax1.plot(x_opt, y_opt, 'm-', label='Traj_Optimize')
ax1.plot(target_position[0], target_position[1], 'go', markersize=10, label='Target')
ax1.plot(r0[0], r0[1], 'ko', markersize=6, label='Start')
ax1.plot(positions[-1][0], positions[-1][1], 'ro', markersize=6, label='Target_Init')
ax1.plot(final_pos[0], final_pos[1], 'mo', markersize=6, label='Target_Optimize')
ax1.grid(True)
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Traj_Compariso ')
ax1.legend()

# 绘制优化历史
iterations = range(len(loss_history))
ax2.plot(iterations, loss_history, 'r-')
ax2.set_xlabel('Interatoin')
ax2.set_ylabel('loss）')
ax2.set_title('Process')
ax2.grid(True)

plt.tight_layout()
plt.show()

# 创建优化过程的动画
fig3, ax3 = plt.subplots(figsize=(10, 8))

opt_line, = ax3.plot([], [], 'b-', lw=1, alpha=0.5)
opt_point, = ax3.plot([], [], 'bo', markersize=8)
target_point2, = ax3.plot([target_position[0]], [target_position[1]], 'go', markersize=10, label='目标')
start_point, = ax3.plot([r0[0]], [r0[1]], 'ko', markersize=6, label='Start')

ax3.set_xlabel('X Position (m)')
ax3.set_ylabel('Y Position (m)')
ax3.set_title('Optimization_Visualization')
ax3.grid(True)

# 计算每次优化对应的轨迹
optimization_trajectories = []
for v in opt_history:
    traj = r0 + v * t_col + 0.5 * a * t_col**2
    optimization_trajectories.append(traj)

# 设置图表范围
all_x = np.concatenate([traj[:, 0] for traj in optimization_trajectories] + [np.array([target_position[0]])])
all_y = np.concatenate([traj[:, 1] for traj in optimization_trajectories] + [np.array([target_position[1]])])
margin_x = (all_x.max() - all_x.min()) * 0.1 or 1
margin_y = (all_y.max() - all_y.min()) * 0.1 or 1
ax3.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
ax3.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

iter_template = 'Iteration: %d'
iter_text = ax3.text(0.05, 0.95, '', transform=ax3.transAxes)
loss_val_template = 'Loss: %.4f'
loss_val_text = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)
velocity_template = 'Velocity: [%.2f, %.2f]'
velocity_text = ax3.text(0.05, 0.85, '', transform=ax3.transAxes)

# 减少轨迹数以加快动画
step = max(1, len(optimization_trajectories) // 50)
selected_trajectories = optimization_trajectories[::step]
selected_losses = loss_history[::step]
selected_velocities = opt_history[::step]

def init_optimization():
    opt_line.set_data([], [])
    opt_point.set_data([], [])
    iter_text.set_text('')
    loss_val_text.set_text('')
    velocity_text.set_text('')
    return opt_line, opt_point, iter_text, loss_val_text, velocity_text

def update_optimization(frame):
    if frame < len(selected_trajectories):
        traj = selected_trajectories[frame]
        opt_line.set_data(traj[:, 0], traj[:, 1])
        opt_point.set_data([traj[-1, 0]], [traj[-1, 1]])
        iter_text.set_text(iter_template % (frame * step))
        loss_val_text.set_text(loss_val_template % selected_losses[frame])
        velocity_text.set_text(velocity_template % (selected_velocities[frame][0], selected_velocities[frame][1]))
    return opt_line, opt_point, iter_text, loss_val_text, velocity_text

ax3.legend()
opt_ani = animation.FuncAnimation(fig3, update_optimization, frames=len(selected_trajectories),
                                init_func=init_optimization, blit=True, interval=100, repeat=True)

plt.show()

# 最终优化轨迹动画
fig4, ax4 = plt.subplots(figsize=(10, 8))

final_line, = ax4.plot([], [], 'b-', lw=2, label='Traj_Init')
final_opt_line, = ax4.plot([], [], 'm-', lw=2, label='Traj_Optimize')
final_point, = ax4.plot([], [], 'ro', markersize=8)
final_target, = ax4.plot([target_position[0]], [target_position[1]], 'go', markersize=10, label='目标')

ax4.set_xlabel('X Position (m)')
ax4.set_ylabel('Y Position (m)')
ax4.set_title('Comparison')
ax4.grid(True)

# 设置图表范围
all_x = np.concatenate([x_traj, x_opt, np.array([target_position[0]])])
all_y = np.concatenate([y_traj, y_opt, np.array([target_position[1]])])
margin_x = (all_x.max() - all_x.min()) * 0.1 or 1
margin_y = (all_y.max() - all_y.min()) * 0.1 or 1
ax4.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
ax4.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

final_time_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes)

def init_final():
    final_line.set_data([], [])
    final_opt_line.set_data([], [])
    final_point.set_data([], [])
    final_time_text.set_text('')
    return final_line, final_opt_line, final_point, final_time_text

def update_final(frame):
    current_t = t_values[frame]
    final_line.set_data(x_traj[:frame+1], y_traj[:frame+1])
    final_opt_line.set_data(x_opt[:frame+1], y_opt[:frame+1])
    
    # 同时在两条轨迹上显示当前位置
    x_current = x_opt[frame]
    y_current = y_opt[frame]
    final_point.set_data([x_current], [y_current])
    
    final_time_text.set_text(time_template % current_t)
    return final_line, final_opt_line, final_point, final_time_text

ax4.legend()
final_ani = animation.FuncAnimation(fig4, update_final, frames=num_frames,
                                  init_func=init_final, blit=True, interval=dt*1000, repeat=True)

plt.show()