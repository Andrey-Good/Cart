from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

LOWER_EQUILIBRIUM = -1.0
UPPER_EQUILIBRIUM = 1.0


@dataclass(frozen=True)
class CartPendulumParams:
    cart_mass: float
    bob_mass: float
    length: float
    gravity: float
    damping: float = 0.0


@dataclass(frozen=True)
class ControlLaw:
    kx: float
    kv: float
    kangle: float
    komega: float
    max_acceleration: float


@dataclass(frozen=True)
class Trajectory:
    solution: np.ndarray

    @property
    def t(self):
        return self.solution[:, 0]

    @property
    def x(self):
        return self.solution[:, 1]

    @property
    def v(self):
        return self.solution[:, 2]

    @property
    def theta(self):
        return self.solution[:, 3]

    @property
    def omega(self):
        return self.solution[:, 4]


def angle_acceleration(angle, omega, cart_acceleration, params, equilibrium_sign):
    gravity = equilibrium_sign * params.gravity * np.sin(angle)
    coupling = cart_acceleration * np.cos(angle)
    return (gravity - coupling) / params.length - params.damping * omega


def closed_accelerations(angle, omega, params, equilibrium_sign=LOWER_EQUILIBRIUM):
    matrix = np.array(
        [
            [params.cart_mass + params.bob_mass, params.bob_mass * params.length * np.cos(angle)],
            [np.cos(angle), params.length],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            params.bob_mass * params.length * np.sin(angle) * omega**2,
            equilibrium_sign * params.gravity * np.sin(angle) - params.length * params.damping * omega,
        ],
        dtype=float,
    )
    return np.linalg.solve(matrix, rhs)


def control_acceleration(state, law):
    x, v, angle, omega = np.asarray(state, dtype=float)
    raw = law.kx * x + law.kv * v + law.kangle * angle + law.komega * omega
    return float(np.clip(raw, -law.max_acceleration, law.max_acceleration))


def closed_rhs(_, state, params, equilibrium_sign=LOWER_EQUILIBRIUM):
    x, v, angle, omega = np.asarray(state, dtype=float)
    cart_acceleration, angle_acc = closed_accelerations(angle, omega, params, equilibrium_sign)
    return np.array([v, cart_acceleration, omega, angle_acc], dtype=float)


def control_rhs(_, state, params, law, equilibrium_sign=UPPER_EQUILIBRIUM):
    x, v, angle, omega = np.asarray(state, dtype=float)
    cart_acceleration = control_acceleration(state, law)
    angle_acc = angle_acceleration(angle, omega, cart_acceleration, params, equilibrium_sign)
    return np.array([v, cart_acceleration, omega, angle_acc], dtype=float)


def rk4fixed(rhs, init, t0, t1, steps):
    init = np.asarray(init, dtype=float)
    if steps <= 0:
        raise ValueError("steps must be positive")

    time = np.linspace(t0, t1, steps + 1, dtype=float)

    solution = solve_ivp(
        rhs,
        (t0, t1),
        init,
        t_eval=time,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )
    return np.column_stack((solution.t, solution.y.T))


def solve_closed(params, init, t_max, steps, equilibrium_sign=LOWER_EQUILIBRIUM):
    rhs = lambda time, state: closed_rhs(time, state, params, equilibrium_sign)
    return Trajectory(solution=rk4fixed(rhs, init, 0.0, t_max, steps))


def solve_control(params, law, init, t_max, steps, equilibrium_sign=UPPER_EQUILIBRIUM):
    rhs = lambda time, state: control_rhs(time, state, params, law, equilibrium_sign)
    return Trajectory(solution=rk4fixed(rhs, init, 0.0, t_max, steps))


def closed_momentum(trajectory, params):
    return (params.cart_mass + params.bob_mass) * trajectory.v + (
        params.bob_mass * params.length * np.cos(trajectory.theta) * trajectory.omega
    )


def control_history(trajectory, law):
    states = trajectory.solution[:, 1:]
    return np.array([control_acceleration(state, law) for state in states], dtype=float)


def cart_and_bob_points(state, params, equilibrium_sign):
    x, _, angle, _ = np.asarray(state, dtype=float)
    bob_x = x + params.length * np.sin(angle)
    bob_y = -equilibrium_sign * params.length * np.cos(angle)
    return x, bob_x, bob_y
