def build_car_scene_xml(walls, out_path="mjx_car_scene.xml", arena_size=8.0):
    wall_geoms = ""
    for i, (x1, y1, x2, y2, r) in enumerate(walls):
        wall_geoms += (
            f'    <geom name="wall_{i}" type="capsule" size="{r}" '
            f'fromto="{x1} {y1} 0.2 {x2} {y2} 0.2" rgba="0.5 0.5 0.5 1"/>\n'
        )

    xml = f"""
<mujoco model="ackermann_car_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="1" cone="pyramidal"/>

  <default>
    <joint damping="0.05"/>
    <geom friction="0.9 0.1 0.1" contype="1" conaffinity="1"/>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="{arena_size} {arena_size} 0.1" rgba="0.3 0.3 0.3 1"/>
{wall_geoms}
    <body name="chassis" pos="0 0 0.1">
      <freejoint name="chassis_free"/>
      <geom name="chassis_geom" type="capsule" size="0.09" fromto="-0.15 0 0 0.15 0 0" mass="3.0" rgba="0.8 0.1 0.1 1"/>

      <body name="wheel_fl" pos="0.1483 0.115 -0.05">
        <joint name="steer_fl" type="hinge" axis="0 0 1" range="-0.6 0.6"/>
        <geom name="knuckle_fl_geom" type="sphere" size="0.01" mass="0.05" rgba="0.2 0.2 0.2 1"/>
        <body name="wheel_fl_spin">
          <joint name="spin_fl" type="hinge" axis="0 1 0"/>
          <geom name="wheel_fl_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
      <body name="wheel_fr" pos="0.1483 -0.115 -0.05">
        <joint name="steer_fr" type="hinge" axis="0 0 1" range="-0.6 0.6"/>
        <geom name="knuckle_fr_geom" type="sphere" size="0.01" mass="0.05" rgba="0.2 0.2 0.2 1"/>
        <body name="wheel_fr_spin">
          <joint name="spin_fr" type="hinge" axis="0 1 0"/>
          <geom name="wheel_fr_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
      <body name="wheel_rl" pos="-0.1483 0.115 -0.05">
        <joint name="spin_rl" type="hinge" axis="0 1 0"/>
        <geom name="wheel_rl_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_rr" pos="-0.1483 -0.115 -0.05">
        <joint name="spin_rr" type="hinge" axis="0 1 0"/>
        <geom name="wheel_rr_geom" type="capsule" size="0.05" fromto="0 -0.0215 0 0 0.0215 0" mass="0.1" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <joint joint1="steer_fl" joint2="steer_fr"/>
  </equality>

  <actuator>
    <motor name="steer" joint="steer_fl" gear="1" ctrlrange="-1 1"/>
    <motor name="throttle_rl" joint="spin_rl" gear="0.35" ctrlrange="-1 1"/>
    <motor name="throttle_rr" joint="spin_rr" gear="0.35" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


def default_walls():
    return [
        (2.0, -2.0, 2.0, 2.0, 0.1),
        (-2.0, 1.0, 1.0, 1.0, 0.1),
        (-1.0, -3.0, -1.0, -1.0, 0.1),
        (3.0, 3.0, 5.0, 3.0, 0.1),
    ]
