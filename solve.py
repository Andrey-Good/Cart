import numpy as np

from pendulum import (
    LOWER_EQUILIBRIUM,
    UPPER_EQUILIBRIUM,
    CartPendulumParams,
    ControlLaw,
    closed_momentum,
    control_history,
    solve_closed,
    solve_control,
)

PARAMS = CartPendulumParams(
    cart_mass=1.0,
    bob_mass=0.25,
    length=1.0,
    gravity=9.81,
    damping=0.25,
)
CONTROL = ControlLaw(
    kx=0.765,
    kv=1.962,
    kangle=28.325,
    komega=8.962,
    max_acceleration=20.0,
)
T_MAX = 20.0
STEPS = int(T_MAX * 60)


def build_closed_case():
    init = [0.0, 0.0, 0.8, 0.0]
    trajectory = solve_closed(
        params=PARAMS,
        init=init,
        t_max=T_MAX,
        steps=STEPS,
        equilibrium_sign=LOWER_EQUILIBRIUM,
    )
    momentum = closed_momentum(trajectory, PARAMS)
    return {
        "name": "Closed cart-pendulum system",
        "mode": "closed",
        "params": PARAMS,
        "trajectory": trajectory,
        "equilibrium_sign": LOWER_EQUILIBRIUM,
        "init": np.asarray(init, dtype=float),
        "momentum": momentum,
        "graph_values": trajectory.x,
        "graph_label": "x(t)",
        "t_max": T_MAX,
        "steps": STEPS,
    }


def build_control_case():
    init = [0.0, 0.0, 0.25, 0.0]
    trajectory = solve_control(
        params=PARAMS,
        law=CONTROL,
        init=init,
        t_max=T_MAX,
        steps=STEPS,
        equilibrium_sign=UPPER_EQUILIBRIUM,
    )
    control = control_history(trajectory, CONTROL)
    return {
        "name": "Inverted pendulum control on a cart",
        "mode": "control",
        "params": PARAMS,
        "trajectory": trajectory,
        "equilibrium_sign": UPPER_EQUILIBRIUM,
        "init": np.asarray(init, dtype=float),
        "control": control,
        "graph_values": trajectory.theta,
        "graph_label": "phi(t)",
        "t_max": T_MAX,
        "steps": STEPS,
        "law": CONTROL,
    }


def print_closed_case(case):
    trajectory = case["trajectory"]
    momentum = case["momentum"]
    print("1) Closed cart-pendulum system:")
    print("   solution = rk4fixed(F, init, 0, t_max, steps)")
    print("   state u = (x, v, theta, omega)")
    print(f"   init = ({case['init'][0]:.3f}, {case['init'][1]:.3f}, {case['init'][2]:.3f}, {case['init'][3]:.3f})")
    print(f"   x range: {trajectory.x.min():.6f} .. {trajectory.x.max():.6f} m")
    print(f"   theta range: {trajectory.theta.min():.6f} .. {trajectory.theta.max():.6f} rad")
    print(f"   max |v|: {np.max(np.abs(trajectory.v)):.6f} m/s")
    print(f"   max |omega|: {np.max(np.abs(trajectory.omega)):.6f} rad/s")
    print(f"   momentum drift: {np.max(np.abs(momentum - momentum[0])):.2e}")


def print_control_case(case):
    trajectory = case["trajectory"]
    control = case["control"]
    after_five_seconds = np.searchsorted(trajectory.t, 5.0)
    print()
    print("2) Inverted pendulum control on a cart:")
    print("   solution = rk4fixed(F, init, 0, t_max, steps)")
    print("   state u = (x, v, phi, omega)")
    print(f"   init = ({case['init'][0]:.3f}, {case['init'][1]:.3f}, {case['init'][2]:.3f}, {case['init'][3]:.3f})")
    print(
        "   control a = clip(kx*x + kv*v + kp*phi + kd*omega, "
        f"-{case['law'].max_acceleration:.1f}, {case['law'].max_acceleration:.1f})"
    )
    print(f"   final phi: {trajectory.theta[-1]:.6f} rad")
    print(f"   max |phi| after 5 s: {np.max(np.abs(trajectory.theta[after_five_seconds:])):.6f} rad")
    print(f"   max |x|: {np.max(np.abs(trajectory.x)):.6f} m")
    print(f"   max |a|: {np.max(np.abs(control)):.6f} m/s^2")


def main():
    closed_case = build_closed_case()
    control_case = build_control_case()
    print_closed_case(closed_case)
    print_control_case(control_case)


if __name__ == "__main__":
    main()
