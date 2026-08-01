"""
Fetches an aerial/satellite background image (ESRI World Imagery), sized either
to match the glide footprint of a Cessna 172 from a given cruising altitude, or
to an explicit ground area, for use as a scenario background image (see
main.py's BACKGROUND_IMAGE).
"""
import argparse
import math

import requests

from utils import cessna_glide_ratio, density_from_altitude

NM_PER_DEGREE_LAT = 60.0  # exact, by definition of the nautical mile
FT_PER_NM = 6076.12

ESRI_EXPORT_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"


def glide_distance_nm(altitude_ft, weight, airspeed, heading, wind_direction, wind_strength):
    density = density_from_altitude(altitude_ft)
    glide_ratio, _ = cessna_glide_ratio(weight, density, airspeed, heading - wind_direction, wind_strength)
    return altitude_ft * glide_ratio / FT_PER_NM


def bbox_for_area(center_lat, center_lon, width_nm, height_nm):
    half_lat_deg = (height_nm / 2) / NM_PER_DEGREE_LAT
    half_lon_deg = (width_nm / 2) / (NM_PER_DEGREE_LAT * math.cos(math.radians(center_lat)))
    return (
        center_lon - half_lon_deg,
        center_lat - half_lat_deg,
        center_lon + half_lon_deg,
        center_lat + half_lat_deg,
    )


def fetch_image(bbox, width_px, height_px, out_path):
    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{width_px},{height_px}",
        "format": "png",
        "transparent": "false",
        "f": "image",
    }
    response = requests.get(ESRI_EXPORT_URL, params=params, timeout=30)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lat", type=float, help="Center latitude (degrees)")
    parser.add_argument("lon", type=float, help="Center longitude (degrees)")
    sizing = parser.add_mutually_exclusive_group(required=True)
    sizing.add_argument("--altitude", type=float, help="Cruising altitude in feet (4000-10000); image is sized to the glide footprint")
    sizing.add_argument("--width-nm", type=float, help="Explicit ground width to capture, in nautical miles (height follows from the pixel aspect ratio)")
    parser.add_argument("--weight", type=float, default=2400, help="Aircraft weight, lbs (default: 2400)")
    parser.add_argument("--airspeed", type=float, default=67, help="Airspeed, knots (default: 67, best glide speed)")
    parser.add_argument("--heading", type=float, default=0, help="Aircraft heading, degrees (default: 0)")
    parser.add_argument("--wind-direction", type=float, default=0, help="Wind direction, degrees (default: 0)")
    parser.add_argument("--wind-strength", type=float, default=0, help="Wind strength, knots (default: 0)")
    parser.add_argument("--margin", type=float, default=1.1, help="Margin multiplier around glide footprint, only used with --altitude (default: 1.1)")
    parser.add_argument("--width-px", type=int, default=681, help="Output image width in pixels (default: 681)")
    parser.add_argument("--height-px", type=int, default=338, help="Output image height in pixels (default: 338)")
    parser.add_argument("--out", default=None, help="Output path (default: Photos/scenario_<altitude>ft.png or Photos/scenario_<width>nm.png)")
    args = parser.parse_args()

    if args.altitude is not None:
        if not (4000 <= args.altitude <= 10000):
            parser.error("altitude must be between 4000 and 10000 ft")
        distance_nm = glide_distance_nm(
            args.altitude, args.weight, args.airspeed, args.heading, args.wind_direction, args.wind_strength
        )
        width_nm = 2 * distance_nm * args.margin
        out_path = args.out or f"Photos/scenario_{int(args.altitude)}ft.png"
    else:
        width_nm = args.width_nm
        out_path = args.out or f"Photos/scenario_{width_nm:.1f}nm.png"

    height_nm = width_nm * args.height_px / args.width_px

    bbox = bbox_for_area(args.lat, args.lon, width_nm, height_nm)
    fetch_image(bbox, args.width_px, args.height_px, out_path)

    print(f"Saved {out_path}")
    if args.altitude is not None:
        print(f"Glide footprint diameter: {2 * distance_nm:.2f} nm")
    print(f"Image covers: {width_nm:.2f} nm x {height_nm:.2f} nm")
    print(f"Set X_DISTANCE = {width_nm:.2f} in main.py to match this image's scale")


if __name__ == "__main__":
    main()
