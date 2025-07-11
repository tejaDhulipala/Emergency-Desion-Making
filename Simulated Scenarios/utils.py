import math

def cessna_glide_ratio(weight=2400, density=0.002378, airspeed=67, wind_delta: float = 0.0, wind_speed=0):
    """
    Calculate Cessna 172 glide ratio based on weight, air density, and airspeed.
    
    Parameters:
    weight (float): Aircraft weight in pounds (default: 2400 lbs)
    density (float): Air density in slug/ft³ (default: 0.002378 - sea level standard)
    airspeed (float): Airspeed in knots (default: 67 knots)
    wind_delta (float): difference between aircraft heading and wind direction
    wind strength (float): wind strength in knots
    
    Returns:
    float: Adjusted glide ratio
    """
    
    # Standard reference conditions
    GR_0 = 9.0                    # Baseline glide ratio
    W_0 = 2400                    # Standard weight (lbs)
    RHO_0 = 0.002378             # Standard density (slug/ft³)
    V_0 = 67                      # Best glide speed (knots)
    
    # Calculate correction factors
    K_weight = (W_0 / weight) ** 0.5
    K_density = (density / RHO_0) ** 0.25
    K_speed = 1 / (0.5 * (airspeed / V_0) ** 2 + 0.5 * (V_0 / airspeed) ** 2)
    
    # Calculate adjusted glide ratio
    glide_ratio = GR_0 * K_weight * K_density * K_speed
    ground_speed = airspeed - wind_speed * math.cos(wind_delta / 180 * math.pi)
    if wind_speed != 0:
        glide_ratio *= ground_speed / wind_speed 
    
    return glide_ratio, ground_speed

def density_from_altitude(altitude_ft, temp_f=59):
    """
    Calculate air density from altitude using standard atmosphere.
    
    Parameters:
    altitude_ft (float): Altitude in feet
    temp_f (float): Temperature in Fahrenheit (default: 59°F standard)
    
    Returns:
    float: Air density in slug/ft³
    """
    
    # Standard atmosphere constants
    RHO_0 = 0.002378  # Sea level density (slug/ft³)
    TEMP_0 = 518.67   # Sea level temperature (°R)
    LAPSE_RATE = 0.00356  # Temperature lapse rate (°R/ft)
    
    # Convert temp to Rankine
    temp_r = temp_f + 459.67
    
    # Calculate density ratio using standard atmosphere
    temp_ratio = (TEMP_0 - LAPSE_RATE * altitude_ft) / TEMP_0
    density_ratio = temp_ratio ** 4.256
    
    return RHO_0 * density_ratio

def desired_heading(x_cur, y_cur, x_goal, y_goal):
    dx = x_goal - x_cur
    dy = y_goal - y_cur
    angle_rad = math.atan2(dx, dy)  # dx, dy order for compass
    angle_deg = math.degrees(angle_rad)
    compass_heading = (angle_deg + 360) % 360
    return compass_heading

def landing_distance(temperature, headwind, flaps):
    # temperature C
    # headwind knots
    # returns in nm
    assert flaps in [0, 10, 20, 30]
    # Assuming 1000ft pressure altitude
    fifty_ft_tbl = {
        0: 1320,
        10: 1350,
        20: 1385,
        30: 1420,
        40: 1450
    }
    # Round temperature to nearest of 0, 10, 20, 30, or 40
    nearest_temp = min(fifty_ft_tbl.keys(), key=lambda t: abs(temperature - t))
    fifty_ft = fifty_ft_tbl[nearest_temp]
    if headwind < 0:
        if headwind <= -10:
            fifty_ft *= 1.5
        else:
            fifty_ft *= 1 - (1 * headwind / 2 / 10)
    else:
        fifty_ft *= 1 - (headwind / 9 / 10)
    fifty_ft *= 1 + ((30 - flaps) / 30 * 35 / 100)
    return fifty_ft / 6071
    
    

if __name__ == "__main__":
    import random
    def test_random_desired_heading():
        # Generate random current and goal positions
        x_cur = random.randint(0, 100)
        y_cur = random.randint(0, 100)
        x_goal = random.randint(0, 100)
        y_goal = random.randint(0, 100)
        
        heading = desired_heading(x_cur, y_cur, x_goal, y_goal)
        
        print(f"Current position: ({x_cur:.2f}, {y_cur:.2f})")
        print(f"Goal position:    ({x_goal:.2f}, {y_goal:.2f})")
        print(f"Desired heading:  {heading:.2f} degrees")
    
    # Run the test function
    test_random_desired_heading()