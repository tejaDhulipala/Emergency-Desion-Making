import pygame as pg
import sys
import math
from plane import Plane, EnvironmentVariables, Instruction

# --- Visualization parameters ---
BACKGROUND_IMAGE = "Simulated Scenarios/scenario_1.png"
# Set window size to match background image
WIDTH, HEIGHT = 681, 338
PLANE_COLOR = (255, 0, 0)  # Red
TARGET_COLOR = (255, 100, 100)
AFTER_TURN_COLOR = (0, 255, 100)
ARROW_COLOR = (255, 255, 0)
DOT_RADIUS = 8
X_DISTANCE = 10 # X distance in nvm
Y_DISTANCE = int(X_DISTANCE * HEIGHT / WIDTH)
MARGIN = 30  # pixels

# --- Utility for scaling positions ---
def scale(x, y):
    sx = int(x / X_DISTANCE * (WIDTH - 2*MARGIN) + MARGIN)
    sy = int(HEIGHT - MARGIN - y / Y_DISTANCE * (HEIGHT - 2*MARGIN))
    return sx, sy

def draw_ruler(screen, scale, WIDTH, HEIGHT):
    font = pg.font.SysFont(None, 18)
    # X-axis ruler (bottom)
    y_ruler = HEIGHT - MARGIN // 2
    tick_height = 8
    x_tick_interval = max(1, int(X_DISTANCE / 10))
    for x_nm in range(0, X_DISTANCE + 1, x_tick_interval):
        sx, _ = scale(x_nm, 0)
        pg.draw.line(screen, (0, 0, 0), (sx, y_ruler), (sx, y_ruler - tick_height), 2)
        label = font.render(f"{x_nm}", True, (0, 0, 0))
        screen.blit(label, (sx - label.get_width() // 2, y_ruler - tick_height - 2))
    # Y-axis ruler (left)
    x_ruler = MARGIN // 2
    y_tick_interval = max(1, int(Y_DISTANCE / 10))
    for y_nm in range(0, int(Y_DISTANCE) + 1, y_tick_interval):
        _, sy = scale(0, y_nm)
        pg.draw.line(screen, (0, 0, 0), (x_ruler, sy), (x_ruler + tick_height, sy), 2)
        label = font.render(f"{y_nm}", True, (0, 0, 0))
        screen.blit(label, (x_ruler + tick_height + 2, sy - label.get_height() // 2))

def get_instruction_from_input():
    while True:
        try:
            s = input("give instruction (goal_x goal_y airspeed bank_angle flaps forward_slip): ")
            goal_x, goal_y, airspeed, bank_angle, flaps, forward_slip = s.split()
            goal_x = float(goal_x)
            goal_y = float(goal_y)
            airspeed = int(airspeed)
            bank_angle = int(bank_angle)
            flaps = int(flaps)
            forward_slip = forward_slip.lower() == 't'
            return goal_x, goal_y, airspeed, bank_angle, flaps, forward_slip
        except Exception:
            print("Invalid input, try again.")

# --- Main visualization ---
def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Plane Turn Visualization")
    clock = pg.time.Clock()

    # Load background image
    background = pg.image.load(BACKGROUND_IMAGE).convert_alpha()
    background = pg.transform.scale(background, (WIDTH, HEIGHT))

    # Initial plane state
    start_x, start_y = 8, 3
    start_heading = 315  # degrees (north)
    env = EnvironmentVariables(wind_strength=0, wind_direction=0, temperature=15)
    instruction = Instruction(goal_x=10, goal_y=9.6, airspeed=80, bank_angle=30)
    plane = Plane(start_x, start_y, alt=800, airspeed=80, weight=2400, heading=start_heading, env_vars=env, inst=instruction)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        screen.blit(background, (0, 0))
        ox, oy = scale(plane.pos_x, plane.pos_y)
        plane.draw(screen, PLANE_COLOR, ox, oy, heading=plane.heading, radius=DOT_RADIUS, scale=scale)
        tx, ty = scale(plane.instruction.goal_x, plane.instruction.goal_y)
        pg.draw.circle(screen, TARGET_COLOR, (tx, ty), DOT_RADIUS)
        draw_ruler(screen, scale, WIDTH, HEIGHT)
        pg.display.flip()
        clock.tick(30)

        # Get new instruction from user
        if not plane.landing:
            goal_x, goal_y, airspeed, bank_angle, flaps, forward_slip = get_instruction_from_input()
            new_instruction = Instruction(goal_x=goal_x, goal_y=goal_y, airspeed=airspeed, bank_angle=bank_angle, flaps=flaps, forward_slip=forward_slip)
            plane.give_instruction(new_instruction)
            plane.follow_instruction()

    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main() 