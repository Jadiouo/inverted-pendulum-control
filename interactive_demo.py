"""Interactive cart-pole demo (matplotlib UI).

The plant is the full nonlinear cart-pole integrated with RK4; the controller
is a linear state feedback u = -K (X - X_ref) designed on the linearised
model, either by pole placement or by LQR (see cartpole.py).

Controls
    開始 / 暫停        run or pause the simulation
    重置              back to the initial tilt
    推一下 (Kick)      apply a 20 N horizontal force for a few frames
    控制器            cycle: 極點配置 -> LQR gentle -> LQR aggressive -> 無控制
    桿長 L / 目標位置  sliders (the gains are recomputed when L changes)

The info box shows the "mathematical evidence" used in the report: the
controllability rank, open- and closed-loop eigenvalues, K and u.
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from cartpole import (LQR_PRESETS, CartPoleParams, controllability_rank, linearize,
                      lqr_gain, nonlinear_dynamics, pole_placement_gain, rk4_step)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']   # 中文字型
plt.rcParams['axes.unicode_minus'] = False

CONTROLLERS = ["極點配置", "LQR gentle", "LQR aggressive", "無控制"]


def eig_summary(eigs, k=4):
    eigs = np.array(eigs)
    eigs = eigs[np.argsort(np.real(eigs))[::-1]]      # 不穩定的先顯示
    parts = []
    for lam in eigs[:k]:
        a, b = np.real(lam), np.imag(lam)
        parts.append(f"{a:+.3f}" if abs(b) < 1e-8 else f"{a:+.3f}{b:+.3f}j")
    if len(eigs) > k:
        parts.append("...")
    return ", ".join(parts)


class Simulation:
    def __init__(self):
        self.p = CartPoleParams()
        self.x0 = np.array([0.0, 0.0, 0.10, 0.0])
        self.state = self.x0.copy()
        self.target_pos = 0.0
        self.dt = 0.02
        self.running = False
        self.controller = 0                      # index into CONTROLLERS

        self.push_timer = 0
        self.push_force_N = 20.0
        self.push_frames = 10

        self.last_u_control = 0.0
        self.last_u_total = 0.0
        self.t = 0.0
        self.time_history, self.theta_history, self.u_history = [0.0], [self.state[2]], [0.0]
        self.max_history = 250
        self.update_matrices()

    # ------------------------------------------------------------------
    def update_matrices(self, val=None):
        self.A, self.B = linearize(self.p)
        self.ctrl_rank = controllability_rank(self.A, self.B)
        self.eigs_open = np.linalg.eigvals(self.A)
        self.gains = {
            "極點配置": pole_placement_gain(self.A, self.B),
            "LQR gentle": lqr_gain(self.A, self.B, *LQR_PRESETS["gentle"]),
            "LQR aggressive": lqr_gain(self.A, self.B, *LQR_PRESETS["aggressive"]),
            "無控制": np.zeros((1, 4)),
        }
        self.K = self.gains[CONTROLLERS[self.controller]]
        self.eigs_closed = np.linalg.eigvals(self.A - self.B @ self.K)

    def reset(self):
        self.state = self.x0.copy()
        self.t = 0.0
        self.time_history, self.theta_history, self.u_history = [0.0], [self.state[2]], [0.0]
        self.last_u_control = self.last_u_total = 0.0
        self.push_timer = 0

    def step(self):
        if not self.running:
            return
        ref = np.array([self.target_pos, 0.0, 0.0, 0.0])
        u_control = -float((self.K @ (self.state - ref)).item())

        disturbance = 0.0
        if self.push_timer > 0:
            disturbance = self.push_force_N
            self.push_timer -= 1
        u_total = u_control + disturbance
        self.last_u_control, self.last_u_total = u_control, u_total

        self.state = rk4_step(lambda s, u: nonlinear_dynamics(s, u, self.p), self.state, u_total, self.dt)
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi   # 掉下去時角度繞回 [-pi, pi]

        self.t += self.dt
        self.time_history.append(self.t)
        self.theta_history.append(self.state[2])
        self.u_history.append(u_total)
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.theta_history.pop(0)
            self.u_history.pop(0)


# ----------------------------------------------------------------------
sim = Simulation()
fig = plt.figure(figsize=(12, 7))
ax_anim = plt.axes([0.45, 0.45, 0.50, 0.50])
ax_plot = plt.axes([0.45, 0.08, 0.50, 0.30])
ax_controls = plt.axes([0.05, 0.05, 0.35, 0.90])
ax_controls.axis('off')

slider_L = Slider(plt.axes([0.10, 0.85, 0.25, 0.03]), '桿長 L', 0.1, 2.0, valinit=sim.p.L)
slider_Pos = Slider(plt.axes([0.10, 0.75, 0.25, 0.03]), '目標位置', -2.0, 2.0, valinit=sim.target_pos)
btn_start = Button(plt.axes([0.10, 0.62, 0.10, 0.08]), '開始', color='lightgreen', hovercolor='0.9')
btn_reset = Button(plt.axes([0.25, 0.62, 0.10, 0.08]), '重置', color='salmon', hovercolor='0.9')
btn_push = Button(plt.axes([0.10, 0.50, 0.25, 0.08]), '推一下 (Kick)', color='lightblue', hovercolor='0.9')
btn_mode = Button(plt.axes([0.10, 0.38, 0.25, 0.08]), f'控制器：{CONTROLLERS[0]}', color='khaki', hovercolor='0.9')

plt.figtext(0.05, 0.15,
            "操作說明:\n"
            "1. 按下「開始」啟動/暫停\n"
            "2. 拖動「目標位置」讓車子移動\n"
            "3. 按「推一下」施加瞬間外力\n"
            "4. 按「控制器」在極點配置 / LQR / 無控制之間切換\n\n"
            "非線性模型 + RK4 積分；K 由線性化模型設計",
            fontsize=11, ha='left')

info_text = fig.text(0.05, 0.92, "", fontsize=10, va='top')


def refresh_info_box():
    max_re_open = np.max(np.real(sim.eigs_open))
    max_re_closed = np.max(np.real(sim.eigs_closed))
    info = (
        f"[控制器] {CONTROLLERS[sim.controller]}\n"
        f"[狀態空間] X_dot = A X + B u（線性化）\n"
        f"[可控性] rank(C) = {sim.ctrl_rank} / 4\n"
        f"[開迴路] eig(A) = {eig_summary(sim.eigs_open)} | max Re = {max_re_open:+.3f} → "
        f"{'不穩定' if max_re_open > 0 else '穩定'}\n"
        f"[閉迴路] eig(A-BK) = {eig_summary(sim.eigs_closed)} | max Re = {max_re_closed:+.3f} → "
        f"{'不穩定' if max_re_closed > 0 else '穩定'}\n"
        f"[K] {np.array2string(sim.K, precision=3)}\n"
        f"[u] u_control={sim.last_u_control:+.2f} N, u_total={sim.last_u_total:+.2f} N"
    )
    info_text.set_text(info)


def update_params(val):
    sim.p.L = slider_L.val
    sim.target_pos = slider_Pos.val
    sim.update_matrices()
    refresh_info_box()


def toggle_run(event):
    sim.running = not sim.running
    btn_start.label.set_text('暫停' if sim.running else '繼續')


def reset_sim(event):
    sim.running = False
    btn_start.label.set_text('開始')
    slider_Pos.reset()
    sim.reset()
    refresh_info_box()


def push_action(event):
    sim.push_timer = sim.push_frames


def cycle_controller(event):
    sim.controller = (sim.controller + 1) % len(CONTROLLERS)
    sim.update_matrices()
    btn_mode.label.set_text(f'控制器：{CONTROLLERS[sim.controller]}')
    line_theta.set_color('r' if CONTROLLERS[sim.controller] == "無控制" else 'g')
    sim.reset()
    refresh_info_box()


slider_L.on_changed(update_params)
slider_Pos.on_changed(update_params)
btn_start.on_clicked(toggle_run)
btn_reset.on_clicked(reset_sim)
btn_push.on_clicked(push_action)
btn_mode.on_clicked(cycle_controller)

# ----------------------------------------------------------------------
ax_anim.set_xlim(-3, 3)
ax_anim.set_ylim(-1, 3)
ax_anim.set_aspect('equal')
ax_anim.grid(True)
ax_anim.set_title("倒單擺即時模擬")
cart_w, cart_h = 0.6, 0.3
patch_cart = plt.Rectangle((0, 0), cart_w, cart_h, fc='blue', ec='k')
line_pole, = ax_anim.plot([], [], 'r-', lw=4, marker='o')
ax_anim.add_patch(patch_cart)
ax_anim.plot([-10, 10], [0, 0], 'k-', lw=1)

ax_plot.set_title("角度 (phi) 與控制輸入 u 的時間響應")
ax_plot.set_ylabel("phi (rad)")
ax_plot.grid(True)
line_theta, = ax_plot.plot([], [], 'g-', lw=1.8, label='phi(t)')
ax_u = ax_plot.twinx()
ax_u.set_ylabel("u (N)")
line_u, = ax_u.plot([], [], 'b-', lw=1.2, alpha=0.7, label='u(t)')
ax_plot.legend(loc='upper left')
refresh_info_box()


def animate(frame):
    sim.step()
    x, _, theta, _ = sim.state
    patch_cart.set_xy((x - cart_w / 2, 0))
    line_pole.set_data([x, x + sim.p.L * np.sin(theta)], [cart_h, cart_h + sim.p.L * np.cos(theta)])

    line_theta.set_data(sim.time_history, sim.theta_history)
    line_u.set_data(sim.time_history, sim.u_history)
    tmin, tmax = min(sim.time_history), max(sim.time_history)
    ax_plot.set_xlim(tmin, tmax + 0.5)
    th = np.array(sim.theta_history)
    ax_plot.set_ylim(float(np.min(th) - 0.1), float(np.max(th) + 0.1))
    uu = np.array(sim.u_history)
    ax_u.set_ylim(float(np.min(uu) - 5.0), float(np.max(uu) + 5.0))
    refresh_info_box()
    return patch_cart, line_pole, line_theta, line_u


ani = animation.FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)
plt.show()
