"""
diff_ackermann_dynamics.py

Differentiable Ackermann (bicycle-model) car dynamics in PyTorch, built as
a drop-in replacement for GRaD-Nav++'s differentiable quadrotor dynamics
(paper: https://arxiv.org/pdf/2506.14009, Section II-A2).

The paper's quadrotor dynamics take a control input u_t = (body-rate
command, thrust command) and integrate quaternion-based rigid body
dynamics with a PD-style low-level rate controller:

    w_dot = I^-1 [ Kp(w_d - w) - Kd*w_dot - w x (Iw) ]
    q_{t+1} = norm( q_t + (dt/2) q_t (x) [0, w] )
    a = (1/m) R(q) [0 0 T]^T + g,   T = c * T_max

This module replaces that with the equivalent for a ground vehicle: a
control input u_t = (steering command, throttle command), integrated
through bicycle-model kinematics, with a first-order actuator lag on
steering (playing the same structural role as the paper's PD rate
controller -- i.e. commands don't apply instantaneously, they have to
"catch up", which matters for DiffRL since it changes the gradient
landscape the same way the quadrotor's rate controller does).

State vector:  s = [x, y, theta, v, delta]
    x, y      -- position
    theta     -- heading (yaw)
    v         -- forward speed
    delta     -- current (actual, lagged) steering angle

Control vector: u = [delta_cmd, throttle_cmd]
    delta_cmd    -- commanded steering angle, in [-delta_max, delta_max]
    throttle_cmd -- commanded acceleration, in [-1, 1] (scaled by a_max)

All operations are standard differentiable PyTorch ops (sin/cos/tan,
clamps), so gradients flow through step() the same way they would through
the paper's quaternion integration -- this is what makes DiffRL viable
(policy gradients computed by backpropagating through the simulated
trajectory, not just through a scalar reward).
"""

import torch


class DiffAckermannDynamics:
    def __init__(
        self,
        wheelbase: float = 0.29,   # meters -- MUSHR's approx wheelbase (1/10 scale RC chassis)
        delta_max: float = 0.34,   # radians -- max steering angle (~19.5 deg, typical MUSHR limit)
        a_max: float = 3.0,        # m/s^2 -- max acceleration
        v_max: float = 3.0,        # m/s -- max speed
        steering_tau: float = 0.15,  # seconds -- actuator lag time constant for steering
        dt: float = 0.02,          # seconds -- integration timestep
    ):
        self.L = wheelbase
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.tau = steering_tau
        self.dt = dt

    def step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        state:   (..., 5) tensor [x, y, theta, v, delta]
        control: (..., 2) tensor [delta_cmd, throttle_cmd], each in [-1, 1]
        returns: (..., 5) tensor, next state

        Differentiable end-to-end: safe to call repeatedly inside a training
        loop and backpropagate through the whole rollout (DiffRL-style).
        """
        x, y, theta, v, delta = state.unbind(-1)
        delta_cmd_raw, throttle_cmd = control.unbind(-1)

        delta_cmd = torch.clamp(delta_cmd_raw, -1.0, 1.0) * self.delta_max
        throttle = torch.clamp(throttle_cmd, -1.0, 1.0) * self.a_max

        # first-order actuator lag on steering -- steering doesn't snap
        # instantly to the commanded angle, it moves toward it. This is
        # the Ackermann-model analogue of the paper's PD body-rate
        # controller: it means control inputs have smooth, differentiable
        # dynamics rather than being applied as instantaneous state jumps.
        delta_dot = (delta_cmd - delta) / self.tau
        delta_next = delta + delta_dot * self.dt
        delta_next = torch.clamp(delta_next, -self.delta_max, self.delta_max)

        # bicycle-model kinematics (rear-axle reference point)
        theta_dot = (v / self.L) * torch.tan(delta)
        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        v_dot = throttle

        x_next = x + x_dot * self.dt
        y_next = y + y_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        v_next = torch.clamp(v + v_dot * self.dt, -self.v_max, self.v_max)

        return torch.stack([x_next, y_next, theta_next, v_next, delta_next], dim=-1)

    def rollout(self, state0: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        state0:   (..., 5) initial state
        controls: (..., T, 2) sequence of T controls
        returns:  (..., T+1, 5) state trajectory including the initial state

        This is the piece DiffRL actually needs: a multi-step rollout that
        stays differentiable end-to-end, so a loss computed on the final
        (or intermediate) states can backpropagate all the way through
        every control input in the sequence.
        """
        states = [state0]
        s = state0
        for t in range(controls.shape[-2]):
            s = self.step(s, controls[..., t, :])
            states.append(s)
        return torch.stack(states, dim=-2)
