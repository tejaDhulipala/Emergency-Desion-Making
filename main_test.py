"""Exercise main.py's dynamic-background pipeline (Plane physics + fetch_photo) with a
scripted sequence of instructions, without running pygame or typing anything interactively.
Just open the saved PNGs in CurPhoto/ afterward to eyeball the result."""

import os
from plane import Plane, EnvironmentVariables, Instruction
from map import SatelliteMap
import main


def run(origin_lat=28.106733, origin_lon=-80.679769, alt=8000, airspeed=80, weight=2400,
        heading=270, env_vars=None):
    # origin_lat/origin_lon default: 28°06'22.75"N 80°41'15.89"W
    if env_vars is None:
        env_vars = EnvironmentVariables(wind_strength=0, wind_direction=0, temperature=15)

    os.makedirs(main.CUR_PHOTO_DIR, exist_ok=True)
    smap = SatelliteMap()
    plane = Plane(main.START_X, main.START_Y, alt=alt, airspeed=airspeed, weight=weight, heading=heading, env_vars=env_vars, inst=None)

    scripted_instructions = [
        Instruction(goal_x=2, goal_y=6.6, airspeed=80, bank_angle=30),
        Instruction(goal_x=2.5, goal_y=6.8, airspeed=75, bank_angle=25),
        Instruction(goal_x=3, goal_y=7, airspeed=70, bank_angle=20, flaps=10),
        Instruction(goal_x=3.2, goal_y=7.1, airspeed=65, bank_angle=15, flaps=20),
    ]

    for i, instr in enumerate(scripted_instructions):
        path, size_nm, lat, lon = main.fetch_photo(smap, plane, i, origin_lat, origin_lon)
        print(f"turn {i}: pos=({plane.pos_x:.2f},{plane.pos_y:.2f}) alt={plane.alt:.0f}ft "
              f"lat/lon=({lat:.5f},{lon:.5f}) size_nm={size_nm:.3f} -> {path}")
        if plane.landing:
            break
        plane.give_instruction(instr)
        plane.follow_instruction()

    path, size_nm, lat, lon = main.fetch_photo(smap, plane, len(scripted_instructions), origin_lat, origin_lon)
    print(f"final: pos=({plane.pos_x:.2f},{plane.pos_y:.2f}) alt={plane.alt:.0f}ft "
          f"lat/lon=({lat:.5f},{lon:.5f}) size_nm={size_nm:.3f} -> {path}")


if __name__ == "__main__":
    run()
