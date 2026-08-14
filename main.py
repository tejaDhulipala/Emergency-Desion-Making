import pygame as pg
import sys
import os
from plane import Plane, EnvironmentVariables, Instruction
from map import SatelliteMap, offset_latlon

# --- Visualization parameters ---
WIDTH, HEIGHT = 900, 900  # square, matches SatelliteMap's square output
PLANE_COLOR = (255, 0, 0)  # Red
TARGET_COLOR = (255, 100, 100)
AFTER_TURN_COLOR = (0, 255, 100)
ARROW_COLOR = (255, 255, 0)
DOT_RADIUS = 8
MARGIN = 30  # pixels, ruler label spacing only

# --- Geo / imagery parameters ---
ORIGIN_LAT, ORIGIN_LON = 28.106733, -80.679769  # 28°06'22.75"N 80°41'15.89"W
START_X, START_Y = 0, 0  # plane's local (pos_x, pos_y) at ORIGIN_LAT/ORIGIN_LON
GLIDE_RATIO = 9  # Cessna 172 nominal glide ratio
FT_PER_NM = 6076.12
MIN_SIZE_FT = 200  # floor so size_nm never collapses to ~0 near landing
CUR_PHOTO_DIR = "CurPhoto"

# --- Utility for scaling positions: always centers (center_x, center_y) on screen ---
def make_scale(center_x, center_y, size_nm):
    def scale(x, y):
        sx = int(WIDTH / 2 + (x - center_x) / size_nm * WIDTH)
        sy = int(HEIGHT / 2 - (y - center_y) / size_nm * HEIGHT)
        return sx, sy
    return scale

GRID_LINE_COLOR = (255, 255, 255)
GRID_LABEL_COLOR = (255, 0, 0)
N_SIDE_TICKS = 8  # ticks above/below (and left/right of) center, so 2*N_SIDE_TICKS+1 total per axis
CORNER_CLEARANCE = 45  # px; skip a label if it would collide with the other axis's labels in the corner

def draw_ruler(screen, scale, size_nm, center_x, center_y):
    font = pg.font.SysFont(None, 16)
    step = (size_nm / 2) / N_SIDE_TICKS
    y_label_row = HEIGHT - MARGIN // 2
    x_label_col = MARGIN // 2

    ticks = []
    grid_surface = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
    for k in range(-N_SIDE_TICKS, N_SIDE_TICKS + 1):
        x_nm = center_x + k * step
        sx, _ = scale(x_nm, center_y)
        pg.draw.line(grid_surface, (*GRID_LINE_COLOR, 90), (sx, 0), (sx, HEIGHT), 1)

        y_nm = center_y + k * step
        _, sy = scale(center_x, y_nm)
        pg.draw.line(grid_surface, (*GRID_LINE_COLOR, 90), (0, sy), (WIDTH, sy), 1)

        ticks.append((sx, x_nm, sy, y_nm))
    screen.blit(grid_surface, (0, 0))

    for sx, x_nm, sy, y_nm in ticks:
        if sx > x_label_col + CORNER_CLEARANCE:
            label = font.render(f"{x_nm - center_x:+.1f}", True, GRID_LABEL_COLOR)
            screen.blit(label, (sx - label.get_width() // 2, y_label_row))
        if sy < y_label_row - CORNER_CLEARANCE:
            label = font.render(f"{y_nm - center_y:+.1f}", True, GRID_LABEL_COLOR)
            screen.blit(label, (x_label_col, sy - label.get_height() // 2))

def get_instruction_from_input(plane):
    """Prompts for a goal relative to the plane's current position (nm, east/north positive),
    then converts it to the absolute goal_x/goal_y that Plane/Instruction operate on."""
    s = ""
    while True:
        try:
            s = input("give instruction (rel_goal_x rel_goal_y airspeed bank_angle flaps forward_slip): ")
            rel_goal_x, rel_goal_y, airspeed, bank_angle, flaps, forward_slip = s.split()
            goal_x = plane.pos_x + float(rel_goal_x)
            goal_y = plane.pos_y + float(rel_goal_y)
            airspeed = int(airspeed)
            bank_angle = int(bank_angle)
            flaps = int(flaps)
            forward_slip = forward_slip.lower() == 't'
            return goal_x, goal_y, airspeed, bank_angle, flaps, forward_slip, True
        except Exception:
            if s.strip().lower() in ["q", "quit"]:
                return None, None, None, None, None, None, False
            print("Invalid input, try again.")

def fetch_photo(smap, plane, photo_index):
    """Fetch a fresh satellite photo centered on the plane, sized to its glide-reachable radius.
    Pure PIL/requests logic, no pygame dependency, so it's testable without a display."""
    lat, lon = offset_latlon(ORIGIN_LAT, ORIGIN_LON, plane.pos_x - START_X, plane.pos_y - START_Y)
    size_nm = max(plane.alt * GLIDE_RATIO * 2, MIN_SIZE_FT) / FT_PER_NM
    img = smap.get_image(lat, lon, size_nm, out_size=WIDTH)
    path = os.path.join(CUR_PHOTO_DIR, f"{photo_index:04d}.png")
    img.save(path)
    return path, size_nm, lat, lon

def load_background(path):
    surface = pg.image.load(path).convert_alpha()
    return pg.transform.scale(surface, (WIDTH, HEIGHT))

# --- Main visualization ---
def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Plane Turn Visualization")
    clock = pg.time.Clock()

    os.makedirs(CUR_PHOTO_DIR, exist_ok=True)
    smap = SatelliteMap()
    photo_index = 0

    # Initial plane state
    start_heading = 270  # degrees (north)
    env = EnvironmentVariables(wind_strength=0, wind_direction=0, temperature=15)
    plane = Plane(START_X, START_Y, alt=1000, airspeed=80, weight=2400, heading=start_heading, env_vars=env, inst=None)
    print(f"Altitude: {plane.alt}. Weight: {plane.weight}. Heading: {plane.heading}. StartX: {plane.pos_x}. StartY: {plane.pos_y}")
    print(f"Winds {round(env.wind_direction / 10)} @ {env.wind_strength}. Temperature {env.temperature}")

    photo_path, size_nm, _, _ = fetch_photo(smap, plane, photo_index)
    background = load_background(photo_path)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        scale = make_scale(plane.pos_x, plane.pos_y, size_nm)
        screen.blit(background, (0, 0))
        ox, oy = scale(plane.pos_x, plane.pos_y)
        if plane.instruction is not None:
            tx, ty = scale(plane.instruction.goal_x, plane.instruction.goal_y)
            pg.draw.circle(screen, TARGET_COLOR, (tx, ty), DOT_RADIUS)
        plane.draw(screen, PLANE_COLOR, ox, oy, heading=plane.heading, radius=DOT_RADIUS, scale=scale)
        draw_ruler(screen, scale, size_nm, plane.pos_x, plane.pos_y)
        pg.display.flip()
        clock.tick(30)

        # Get new instruction from user
        if not plane.landing and running:
            goal_x, goal_y, airspeed, bank_angle, flaps, forward_slip, running = get_instruction_from_input(plane)
            if not running:
                continue
            new_instruction = Instruction(goal_x=goal_x, goal_y=goal_y, airspeed=airspeed, bank_angle=bank_angle, flaps=flaps, forward_slip=forward_slip)
            plane.give_instruction(new_instruction)
            plane.follow_instruction()
            photo_index += 1
            photo_path, size_nm, _, _ = fetch_photo(smap, plane, photo_index)
            background = load_background(photo_path)

    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main()
