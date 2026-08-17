"""
Mostly sanity checking (not real tests) for utils.py.
"""

from utils import cessna_glide_ratio


def sweep():
    weights = [1800, 2100, 2400]
    # Densities in standard atmosphere at SL, 5000 and 10,000 ft
    # Source: https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
    densities = [0.002378, 20.48 * 10 ** -4, 17.56 * 10 ** -4]
    airspeeds = [50, 65, 90]
    wind_deltas = [0, 90, 180]
    wind_speeds = [0, 10]

    for weight in weights:
        for density in densities:
            for airspeed in airspeeds:
                for wind_delta in wind_deltas:
                    for wind_speed in wind_speeds:
                        glide_ratio, ground_speed = cessna_glide_ratio(
                            weight, density, airspeed, wind_delta, wind_speed
                        )
                        print(
                            f"weight={weight} density={density} airspeed={airspeed} "
                            f"wind_delta={wind_delta} wind_speed={wind_speed} "
                            f"-> glide_ratio={glide_ratio:.3f} ground_speed={ground_speed:.2f}"
                        )


def test_standard_conditions():
    glide_ratio, _ = cessna_glide_ratio(2400, 0.002378, 65, 0.0, 0)
    passed = abs(glide_ratio - 9.0) < 1e-9
    print(f"standard_conditions: {passed}")


def test_tailwind_increases_glide_ratio():
    baseline, _ = cessna_glide_ratio(2400, 0.002378, 65, 0.0, 0)
    tailwind, _ = cessna_glide_ratio(2400, 0.002378, 65, 180, 10)
    passed = tailwind > baseline
    print(f"tailwind_increases_glide_ratio: {passed}")


def test_crosswind_no_change():
    baseline, _ = cessna_glide_ratio(2400, 0.002378, 65, 0.0, 0)
    crosswind, _ = cessna_glide_ratio(2400, 0.002378, 65, 90, 10)
    passed = abs(crosswind - baseline) < 1e-9
    print(f"crosswind_no_change: {passed}")


def test_headwind_decreases_glide_ratio():
    baseline, _ = cessna_glide_ratio(2400, 0.002378, 65, 0.0, 0)
    headwind, _ = cessna_glide_ratio(2400, 0.002378, 65, 0, 10)
    passed = headwind < baseline
    print(f"headwind_decreases_glide_ratio: {passed}")


if __name__ == "__main__":
    sweep()
    test_standard_conditions()
    test_tailwind_increases_glide_ratio()
    test_crosswind_no_change()
    test_headwind_decreases_glide_ratio()
