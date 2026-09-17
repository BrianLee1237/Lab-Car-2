"""
mjx_room_scene.py

An enclosed room for MuJoCo/MJX: real 3D box walls around the perimeter
and inside, plus human obstacles, with the same Ackermann car.

mjx_car_scene builds free-floating capsule "sticks" 0.2m tall on an
unbounded plane -- the car could simply drive out of the obstacle field,
and most straight paths across the map hit nothing, so the lidar had
little to perceive. Here the space is a closed room the car cannot
leave, the walls are full-height boxes rather than low rails, and the
humans are cylinders of roughly person size and footprint, so a lidar
scan returns the kind of structure a real scan would.

Geometry is kept axis-aligned so walls need no quaternions and so the
2D sensing/collision math (mjx_lidar, mjx_obstacle_dist) stays exact:
a box wall of thickness t is, in the plane, the segment along its
centreline inflated by t/2, and a human is a circle.
"""

STEER_RANGE = 0.6  # rad; matches the steer joints' mechanical range

WALL_HEIGHT = 1.0       # m, full height -- tall enough to block the lidar
WALL_THICKNESS = 0.15   # m, full thickness
HUMAN_RADIUS = 0.25     # m, body radius (~0.5m shoulder width)
HUMAN_HEIGHT = 1.70     # m


def build_room_scene_xml(walls, humans, out_path="mjx_room_scene.xml",
                          room_half=8.0, floor_pad=2.0):
    """walls: list of (x1, y1, x2, y2, half_thickness) axis-aligned segments.
    humans: list of (x, y, radius)."""
    geoms = ""

    for i, (x1, y1, x2, y2, r) in enumerate(walls):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        hx = max(abs(x2 - x1) / 2.0, r)
        hy = max(abs(y2 - y1) / 2.0, r)
        geoms += (
            f'    <geom name="wall_{i}" type="box" '
            f'size="{hx:.4f} {hy:.4f} {WALL_HEIGHT/2:.4f}" '
            f'pos="{cx:.4f} {cy:.4f} {WALL_HEIGHT/2:.4f}" '
            f'rgba="0.55 0.55 0.60 1"/>\n'
        )

    # Humans are MOCAP bodies so they can be walked kinematically: their
    # pose is written each step via data.mocap_pos, and they collide with
    # the car without the car being able to knock them over or push them
    # off course -- which is how a person behaves next to a small RC car.
    # Capsules, not cylinders: MJX has no cylinder-box collision, and a
    # vertical capsule presents the same circular footprint to a planar
    # lidar anyway.
    for i, (hx_, hy_, hr) in enumerate(humans):
        half_h = HUMAN_HEIGHT * 0.39          # capsule half-length
        geoms += (
            f'    <body name="human_{i}" mocap="true" '
            f'pos="{hx_:.4f} {hy_:.4f} {half_h + hr:.4f}">\n'
            f'      <geom name="human_geom_{i}" type="capsule" '
            f'size="{hr:.3f} {half_h:.3f}" rgba="0.85 0.55 0.35 1"/>\n'
            f'    </body>\n'
        )

    floor = room_half + floor_pad
    xml = f"""
<mujoco model="ackermann_car_room">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="4" cone="pyramidal"/>

  <default>
    <joint damping="0.05"/>
    <geom friction="2.0 0.1 0.1" contype="1" conaffinity="1"/>
  </default>

  <worldbody>
    <light name="ceiling" pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="{floor} {floor} 0.1" rgba="0.32 0.33 0.35 1"/>
{geoms}
    <body name="chassis" pos="0 0 0.1">
      <freejoint name="chassis_free"/>
      <geom name="chassis_geom" type="capsule" size="0.09" fromto="-0.15 0 0 0.15 0 0" mass="3.0" rgba="0.8 0.1 0.1 1"/>

      <body name="wheel_fl" pos="0.1483 0.115 -0.05">
        <joint name="steer_fl" type="hinge" axis="0 0 1" range="-{STEER_RANGE} {STEER_RANGE}" limited="true" damping="1.0"/>
        <geom name="knuckle_fl_geom" type="sphere" size="0.01" mass="0.05" rgba="0.2 0.2 0.2 1"/>
        <body name="wheel_fl_spin">
          <joint name="spin_fl" type="hinge" axis="0 1 0" damping="0.001"/>
          <geom name="wheel_fl_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
      <body name="wheel_fr" pos="0.1483 -0.115 -0.05">
        <joint name="steer_fr" type="hinge" axis="0 0 1" range="-{STEER_RANGE} {STEER_RANGE}" limited="true" damping="1.0"/>
        <geom name="knuckle_fr_geom" type="sphere" size="0.01" mass="0.05" rgba="0.2 0.2 0.2 1"/>
        <body name="wheel_fr_spin">
          <joint name="spin_fr" type="hinge" axis="0 1 0" damping="0.001"/>
          <geom name="wheel_fr_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
      <body name="wheel_rl" pos="-0.1483 0.115 -0.05">
        <joint name="spin_rl" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="wheel_rl_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_rr" pos="-0.1483 -0.115 -0.05">
        <joint name="spin_rr" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="wheel_rr_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <joint joint1="steer_fl" joint2="steer_fr"/>
  </equality>

  <actuator>
    <!-- Steering is a POSITION servo (commanded angle), not a torque
         motor. It was <motor gear="1" ctrlrange="-1 1"/>, which applies
         TORQUE to the steering hinge: the near-inertialess knuckle just
         integrated that torque straight into the joint limit, so the
         command had almost no proportional authority. Measured directly
         (steer_check.py): steer commands spanning 0.05 to 1.0 -- a 20x
         range -- all settled to essentially the same steering angle
         (0.79-0.90 rad) and the same turn radius (0.52m vs 0.47m), and
         the angle blew past the +-0.6 rad mechanical range because a
         soft limit constraint cannot hold against that torque. The
         policy therefore had effectively BINARY steering (hard left /
         hard right) with no way to steer gently -- which is why
         trajectories curved hard, spiralled, overshot, and could never
         make a fine final approach, and why even a proportional oracle
         controller only reached 40%: its computed steering angle was
         being ignored. The real MuSHR car steers with a servo commanded
         to an angle, so a position actuator is also the more faithful
         model. ctrlrange is in radians and matches the joint range;
         callers scale their [-1,1] action by STEER_RANGE. -->
    <position name="steer" joint="steer_fl" kp="80" ctrlrange="-{STEER_RANGE} {STEER_RANGE}" forcerange="-20 20"/>
    <!-- Root cause of the car's earlier crawl-speed (~0.11-0.21 m/s):
         the <default><joint damping="0.05"/></default> block applied to
         EVERY joint, including the drive wheels' spin joints -- so the
         wheels had a phantom viscous brake resisting spin, capping wheel
         angular velocity (and hence car speed) regardless of available
         torque or tire grip (a slip-ratio diagnostic confirmed slip was
         only ~10%, ruling out a traction/grip problem). Real drivetrains
         have no such damping. Explicit damping="0.001" on each spin_*
         joint below overrides the blanket default. Combined with raising
         QVEL_CLAMP (train_diffmjx_final.py) from 15.0 to 40.0 -- which
         was independently capping wheel angular velocity at 15 rad/s
         (0.75 m/s wheel-surface speed), a leftover from when speeds were
         much lower and NaN-safety was the only concern -- an open-loop
         full-throttle test now reaches 1-3 m/s, matching the real
         MuSHR hardware's actual speed range. gear=0.78 (traction-limited
         torque matching mu=2.0, the real MuSHR tire friction spec from
         racecar.urdf) is unchanged and still correct -- the bottleneck
         was never torque or grip, only this spin-joint damping. -->
    <motor name="throttle_rl" joint="spin_rl" gear="0.78" ctrlrange="-1 1"/>
    <motor name="throttle_rr" joint="spin_rr" gear="0.78" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path
