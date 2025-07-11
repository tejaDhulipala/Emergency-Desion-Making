from dataclasses import dataclass

from pygame.math import clamp
from utils import *

@dataclass
class EnvironmentVariables:
    wind_strength: int # knots
    wind_direction: int # magnetic degrees
    temperature: int # celsius

@dataclass
class Instruction:
    goal_x: float
    goal_y: float
    airspeed: int
    bank_angle: int
    flaps: int = 0
    forward_slip: bool = False
    assert flaps in [0, 10, 20, 30]



class Plane:
    def __init__(self, pos_x, pos_y, alt, airspeed, weight, heading, env_vars: EnvironmentVariables, inst: Instruction):
        self.pos_x = pos_x # nm
        self.pos_y = pos_y # nm
        self.alt = alt # ft
        self.airspeed = airspeed # kias
        self.heading = heading # degrees
        self.engine_rpm = 2300 # rpm
        self.aircraft_condition = "intact"
        self.environment_variables = env_vars
        self.time_simulated = 0
        self.instruction = inst
        self.instruction_completed = True
        self.density = density_from_altitude(self.alt, self.environment_variables.temperature)
        self.weight = weight
        self.glide_ratio, self.ground_speed = cessna_glide_ratio(self.weight, self.density, self.airspeed,
        self.heading - self.environment_variables.wind_direction, self.environment_variables.wind_strength)
        self.landing = None

    def give_instruction(self, instruction: Instruction):
        self.instruction = instruction
        self.instruction_completed = False

    def update_glide_ratio(self):
        self.density = density_from_altitude(self.alt, self.environment_variables.temperature)
        self.glide_ratio, self.ground_speed = cessna_glide_ratio(self.weight, self.density, self.airspeed,
        self.heading - self.environment_variables.wind_direction, self.environment_variables.wind_strength)
        if self.instruction.forward_slip:
            self.glide_ratio -= 4
            self.glide_ratio = clamp(self.glide_ratio, 4, 6)
        elif self.instruction.flaps:
            if self.instruction.flaps in [10, 20]:
                self.glide_ratio -= 1
            else: 
                self.glide_ratio -= 2
        print(f"Glide Ratio: {self.glide_ratio}")


    def follow_instruction(self):
        # Follow instruction by first doing a turn, then going along a heading
        # The first step is doing a turn at the current airspeec to the goal heading
        # Calculate turn radius in feet, then convert to nautical miles
        if self.instruction_completed:
            return
        turn_radius_ft = self.ground_speed ** 2 / (11.26 * math.tan(math.radians(self.instruction.bank_angle)))
        turn_radius = turn_radius_ft / 6076  # nautical miles
        desired_heading_value = desired_heading(self.pos_x, self.pos_y, self.instruction.goal_x, self.instruction.goal_y)
        # Determine shortest turn direction
        ground_speed_nms = self.ground_speed / 3600  # knots to nm/sec
        omega_rad = ground_speed_nms / turn_radius / math.pi * 180  # degrees/sec
        dt = abs(desired_heading_value - self.heading) / omega_rad
        self.alt -= 300 * dt / 60
        print(f"Altitude loss from turn {300 * dt / 60}")
        self.heading = desired_heading_value
        self.update_glide_ratio()
        # Go straight in the desired direction
        distance = ((self.pos_x - self.instruction.goal_x) ** 2 + (self.pos_y - self.instruction.goal_y) ** 2) ** 0.5
        glide_distance = self.glide_ratio * (self.alt - 50) / 6071
        self.instruction_completed = True
        print(f"glide distance {glide_distance} distance {distance}")
        if glide_distance <= distance:
            self.alt -= glide_distance * 6071 / self.glide_ratio
            angle_rad = math.radians(self.heading)
            self.pos_x += glide_distance * math.sin(angle_rad)
            self.pos_y += glide_distance * math.cos(angle_rad)
            self.aircraft_condition = "landed"
            self.initiate_landing()
        else:
            self.alt -= distance * 6071 / self.glide_ratio
            self.pos_x = self.instruction.goal_x
            self.pos_y = self.instruction.goal_y
        print(f"({self.pos_x}nm, {self.pos_y}nm, {self.alt}ft, {self.heading} degreees)")
    
    def initiate_landing(self):
        headwind = self.environment_variables.wind_strength * math.cos((self.environment_variables.wind_strength - self.heading) / 180 * math.pi)
        landing_dist = landing_distance(self.environment_variables.temperature, headwind, self.instruction.flaps)
        print(f"Landing distance: {landing_dist}")
        x_start = self.pos_x
        y_start = self.pos_y
        # Calculate end position after traveling landing_dist nm at current heading
        angle_rad = math.radians(self.heading)
        x_end = x_start + landing_dist * math.sin(angle_rad)
        y_end = y_start + landing_dist * math.cos(angle_rad)
        self.landing = x_start, y_start, x_end, y_end

    def draw(self, surface, color, sx, sy, heading=None, radius=8, scale=None):
        """
        Draw the plane as a circle with an arrow for heading.
        sx, sy: screen coordinates (already scaled)
        heading: if None, use self.heading
        scale: function to convert (x, y) in NM to screen coordinates (sx, sy)
        """
        import pygame as pg
        if heading is None:
            heading = self.heading
        pg.draw.circle(surface, color, (sx, sy), radius)
        # Draw heading arrow
        arrow_length = 40
        angle_rad = math.radians(heading)
        arrow_dx = arrow_length * math.sin(angle_rad)
        arrow_dy = -arrow_length * math.cos(angle_rad)
        pg.draw.line(surface, color, (sx, sy), (sx + int(arrow_dx), sy + int(arrow_dy)), 4)
        # Arrowhead
        head_angle1 = angle_rad + math.radians(150)
        head_angle2 = angle_rad - math.radians(150)
        head1 = (sx + int(arrow_dx + 15 * math.sin(head_angle1)), sy + int(arrow_dy - 15 * math.cos(head_angle1)))
        head2 = (sx + int(arrow_dx + 15 * math.sin(head_angle2)), sy + int(arrow_dy - 15 * math.cos(head_angle2)))
        pg.draw.line(surface, color, (sx + int(arrow_dx), sy + int(arrow_dy)), head1, 4)
        pg.draw.line(surface, color, (sx + int(arrow_dx), sy + int(arrow_dy)), head2, 4)
        # Draw landing path if landed
        if self.aircraft_condition == "landed" and self.landing is not None and scale is not None:
            x_start, y_start, x_end, y_end = self.landing
            start_px, start_py = scale(x_start, y_start)
            end_px, end_py = scale(x_end, y_end)
            pg.draw.line(surface, (0, 0, 0), (start_px, start_py), (end_px, end_py), 4)
        



        