"""Automatically fly a scenario end-to-end using an OpenRouter vision LLM to choose each
instruction. Saves every annotated frame the LLM saw, the full decision trajectory (as JSON),
and a final frame showing the landing rollout path."""

import base64
import io
import json
import math
import os
import time

import pygame as pg
import requests

import main
from plane import Plane, EnvironmentVariables, Instruction
from utils import desired_heading, signed_heading_diff, cessna_glide_ratio, V_GLIDE
from utils.constants import (
    MIN_BANK_ANGLE_DEG, MAX_BANK_ANGLE_DEG, FLAP_GR_PENALTY_PER_10_DEG,
    MIN_AIRSPEED_KT, MAX_AIRSPEED_KT, VALID_FLAP_SETTINGS, FT_PER_NM, OBSTACLE_CLEARANCE_FT
)
from utils.paths import CONE_ALPHA_DEG, turn_radius_ft, turn_center

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-4-31b-it"
MAX_RETRIES = 3

# Reference turn radii at best-glide speed (V_GLIDE, no wind) so the system prompt can give
# the LLM concrete numbers -- actual radius scales with ground_speed^2, so it grows if the
# turn is flown faster than this or with a tailwind component (see check_impossible_turn,
# which computes the real radius for each turn's actual airspeed/wind before it's flown).
_REFERENCE_BANK_ANGLES_DEG = (15, 30, 45)
_REFERENCE_TURN_RADII_NM = {
    bank: turn_radius_ft(V_GLIDE, bank) / FT_PER_NM for bank in _REFERENCE_BANK_ANGLES_DEG
}

SYSTEM_PROMPT = f"""You are the pilot of a Cessna 172 whose engine has failed at altitude. Glide the \
aircraft to the safest possible landing using the satellite image and flight data given each turn.

COORDINATE FRAME
The image is centered on the plane (red dot with a red arrow showing its heading). Red gridlines \
are labeled with distance in nautical miles relative to the plane's CURRENT position: positive x is \
east, negative x is west, positive y is north, negative y is south. The image's extent is exactly the \
plane's maximum possible glide distance at its current altitude -- nothing outside the frame is \
reachable, so don't pick a goal near the edge expecting to reach further next turn.

HOW TO DECIDE EACH TURN
You have no memory of previous turns. This image and the data below are the only \
information available -- there is no record of what you chose last time.

Prefer a nearby intermediate waypoint over aiming directly at the final landing spot, \
unless you are close enough to glide straight in. A good reecomendation is to pick a goal roughly \
1/6 to 1/4 of the image's width away in the direction you want to head -- you will get a \
fresh image and choose again next turn regardless, so there's no benefit to committing \
further ahead than you can actually see clearly. Once the plane reaches {OBSTACLE_CLEARANCE_FT}ft, \
it will commence landing. You will be evaluated on the survivability of the landing path once you touch down based on \
surfaces it touches. 

Manage airspeed, bank angle, and flaps to guide the aircraft to a safe landing.  

TURN MECHANICS
Your target heading is always the straight line from your current position to the goal you choose -- \
you don't set heading directly, only the goal. If the heading change needed to reach your goal is \
{CONE_ALPHA_DEG} degrees or less, the plane glides straight there with only a shallow correction. \
Otherwise, it turns at the bank_angle you specify (must be between {MIN_BANK_ANGLE_DEG} and \
{MAX_BANK_ANGLE_DEG} degrees) until pointed at the goal, then glides straight the rest of the way -- a \
steeper bank turns tighter but costs more altitude per degree turned. Every bank_angle/airspeed \
combination has a turn radius; if the goal you pick falls inside the circle that turn would fly, no \
coordinated turn at that bank angle can ever roll out pointed at it -- pick a goal farther away, or a \
steeper bank_angle to shrink the turn radius. For reference, at {V_GLIDE} kt (best-glide speed) with \
no wind, turn radius is about {_REFERENCE_TURN_RADII_NM[15]:.2f} nm at 15 degrees of bank, \
{_REFERENCE_TURN_RADII_NM[30]:.2f} nm at 30 degrees, and {_REFERENCE_TURN_RADII_NM[45]:.2f} nm at 45 \
degrees.

FLAPS
Flaps reduce glide ratio by {FLAP_GR_PENALTY_PER_10_DEG} for every 10 degrees extended (e.g. 30 \
degrees of flaps costs {3 * FLAP_GR_PENALTY_PER_10_DEG} off the baseline glide ratio).

Respond with ONLY a single JSON object (no markdown fences, no other text) with exactly these keys:
{{"rel_goal_x": <float, nm east(+)/west(-) of current position>,
 "rel_goal_y": <float, nm north(+)/south(-) of current position>,
 "distance_to_goal_nm": <float, straight-line distance to (rel_goal_x, rel_goal_y) -- state this \
explicitly so you can sanity-check it against the guidance above>,
 "airspeed": <int, knots indicated airspeed, between {MIN_AIRSPEED_KT} and {MAX_AIRSPEED_KT}>,
 "bank_angle": <int, degrees, between {MIN_BANK_ANGLE_DEG} and {MAX_BANK_ANGLE_DEG}>,
 "flaps": <int, one of {VALID_FLAP_SETTINGS}>,
 "reasoning": <short string explaining the choice>}}
"""


def load_dotenv(path=".env"):
    values = {}
    if not os.path.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_user_prompt(plane, size_nm, turn, max_turns):
    env = plane.environment_variables
    return (
        f"Turn {turn + 1} of {max_turns} maximum. Altitude: {plane.alt:.0f} ft. "
        f"Heading: {plane.heading:.0f} deg. Airspeed: {plane.airspeed} kt. "
        f"Wind: {env.wind_direction} deg @ {env.wind_strength} kt. Temperature: {env.temperature}C. "
        f"The image spans {size_nm:.2f} nm across, centered on the plane. Choose the next instruction."
    )


def frame_to_data_uri(surface):
    buf = io.BytesIO()
    pg.image.save(surface, buf, "frame.png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def annotate_frame_with_decision(frame_path, scale, goal_x, goal_y, decision):
    """Overlay the LLM's chosen goal (light-red circle) and its airspeed/bank_angle/flaps
    (opaque text, top-left) onto an already-saved frame -- so frame N shows the decision the
    LLM made *after seeing* frame N, not the instruction it was flying on turn N-1."""
    frame = pg.image.load(frame_path).convert_alpha()

    tx, ty = scale(goal_x, goal_y)
    pg.draw.circle(frame, main.TARGET_COLOR, (tx, ty), main.DOT_RADIUS)

    font = pg.font.SysFont(None, 24, bold=True)
    text = (f"airspeed: {decision['airspeed']} kt   bank: {decision['bank_angle']} deg   "
            f"flaps: {decision['flaps']}")
    label = font.render(text, True, (255, 255, 255))
    pad = 4
    backing = pg.Surface((label.get_width() + 2 * pad, label.get_height() + 2 * pad))
    backing.fill((0, 0, 0))
    frame.blit(backing, (10, 10))
    frame.blit(label, (10 + pad, 10 + pad))

    pg.image.save(frame, frame_path)


class InstructionError(Exception):
    """Raised when an LLM decision is malformed or physically invalid. The message is sent
    back to the LLM verbatim (as a user turn) so it can correct itself on the next attempt."""


def call_openrouter(api_key, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.4}
    max_retries = 5
    for attempt in range(max_retries):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429 and attempt < max_retries - 1:
            delay = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            print(f"  (rate limited, retrying in {delay}s...)")
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def parse_instruction_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise InstructionError(
            f"Your response was not valid JSON ({e}). Respond with ONLY a single JSON object, "
            "no markdown fences and no other text."
        ) from e
    if not isinstance(data, dict):
        raise InstructionError("Your response must be a single JSON object, not a list or a scalar.")
    required_keys = ["rel_goal_x", "rel_goal_y", "airspeed", "bank_angle", "flaps"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise InstructionError(
            f"Your JSON object was missing required key(s): {', '.join(missing)}. "
            f"It must include all of: {', '.join(required_keys)}."
        )
    return data


def check_impossible_turn(plane, rel_goal_x, rel_goal_y, airspeed, bank_angle):
    """Raise InstructionError if the chosen goal falls inside the circle the plane's turn
    would fly at this airspeed/bank_angle -- mirrors the check in
    utils.paths._maneuver_altitude_loss, but runs before plane.follow_instruction() so a
    failed attempt never mutates plane state mid-turn."""
    goal_x = plane.pos_x + rel_goal_x
    goal_y = plane.pos_y + rel_goal_y
    target_heading = desired_heading(plane.pos_x, plane.pos_y, goal_x, goal_y)
    delta = signed_heading_diff(plane.heading, target_heading)
    if abs(delta) <= CONE_ALPHA_DEG:
        return  # within the no-turn cone -- glides straight, no turn geometry involved

    turn_sign = 1 if delta >= 0 else -1
    wind_delta = plane.heading - plane.environment_variables.wind_direction
    _, ground_speed = cessna_glide_ratio(plane.weight, plane.density, airspeed, wind_delta,
                                          plane.environment_variables.wind_strength)
    radius_nm = turn_radius_ft(ground_speed, bank_angle) / FT_PER_NM
    center = turn_center((plane.pos_x, plane.pos_y), plane.heading, radius_nm, turn_sign)
    distance_to_center = math.dist((goal_x, goal_y), center)
    if distance_to_center < radius_nm:
        raise InstructionError(
            f"This turn is impossible: your goal needs a heading change of {delta:.0f} degrees, "
            f"outside the {CONE_ALPHA_DEG} degree no-turn cone, so the plane must turn to reach it. "
            f"At {airspeed} kt and {bank_angle} deg bank your turn radius is {radius_nm:.3f} nm, "
            f"but the goal you picked is only {distance_to_center:.3f} nm from the center of that "
            f"turn circle -- it falls inside the circle the plane would fly, so no coordinated turn "
            f"at this bank angle can ever roll out pointed at it. Pick a goal further from your "
            f"current position, increase bank_angle (steeper turn = smaller radius), or" 
            f"decrease speed in order to navigate to the correct point."
        )


def validate_decision(data, plane):
    """Check a parsed decision for out-of-range values, raising a single InstructionError
    describing every problem found (so one retry can fix them all). Only checks the
    turn-radius geometry once every other field is confirmed valid, since it depends on
    a numeric airspeed/bank_angle to compute."""
    errors = []
    rel_goal_x = rel_goal_y = airspeed = bank_angle = flaps = None

    try:
        rel_goal_x = float(data["rel_goal_x"])
    except (TypeError, ValueError):
        errors.append(f"'rel_goal_x' must be a number, got {data['rel_goal_x']!r}.")
    try:
        rel_goal_y = float(data["rel_goal_y"])
    except (TypeError, ValueError):
        errors.append(f"'rel_goal_y' must be a number, got {data['rel_goal_y']!r}.")

    try:
        airspeed = int(round(float(data["airspeed"])))
        if not (MIN_AIRSPEED_KT <= airspeed <= MAX_AIRSPEED_KT):
            errors.append(
                f"'airspeed' must be between {MIN_AIRSPEED_KT} and {MAX_AIRSPEED_KT} knots, "
                f"got {airspeed}."
            )
    except (TypeError, ValueError):
        errors.append(f"'airspeed' must be an integer, got {data['airspeed']!r}.")

    try:
        bank_angle = int(round(float(data["bank_angle"])))
        if not (MIN_BANK_ANGLE_DEG <= bank_angle <= MAX_BANK_ANGLE_DEG):
            errors.append(
                f"'bank_angle' must be between {MIN_BANK_ANGLE_DEG} and {MAX_BANK_ANGLE_DEG} "
                f"degrees, got {bank_angle}."
            )
    except (TypeError, ValueError):
        errors.append(f"'bank_angle' must be an integer, got {data['bank_angle']!r}.")

    try:
        flaps = int(round(float(data["flaps"])))
        if flaps not in VALID_FLAP_SETTINGS:
            errors.append(f"'flaps' must be one of {VALID_FLAP_SETTINGS}, got {flaps}.")
    except (TypeError, ValueError):
        errors.append(f"'flaps' must be an integer, got {data['flaps']!r}.")

    if errors:
        raise InstructionError(" ".join(errors))

    check_impossible_turn(plane, rel_goal_x, rel_goal_y, airspeed, bank_angle)


def build_retry_message(error):
    return (
        f"{error} Respond again with ONLY a corrected single JSON object using the exact "
        "schema described in the system prompt."
    )


def get_llm_instruction(api_key, plane, size_nm, turn, max_turns, image_data_uri):
    user_text = build_user_prompt(plane, size_nm, turn, max_turns)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]},
    ]
    failed_attempts = []
    last_error = None
    for attempt in range(MAX_RETRIES):
        raw = call_openrouter(api_key, messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            data = parse_instruction_json(raw)
            validate_decision(data, plane)
            return data, raw, failed_attempts
        except InstructionError as e:
            last_error = e
            failed_attempts.append({"raw_response": raw, "error": str(e)})
            print(f"  (invalid decision, attempt {attempt + 1}/{MAX_RETRIES}: {e})")
            messages.append({"role": "user", "content": build_retry_message(e)})
    raise RuntimeError(f"Could not get a valid instruction after {MAX_RETRIES} attempts: {last_error}")


def run(run_name=None, max_turns=20, origin_lat=28.106733, origin_lon=-80.679769,
        alt=1500, airspeed=80, weight=2400, heading=270, env_vars=None):
    # origin_lat/origin_lon default: 28°06'22.75"N 80°41'15.89"W
    if env_vars is None:
        env_vars = EnvironmentVariables(wind_strength=8, wind_direction=270, temperature=15)

    dotenv_values = load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values.get("OPENROUTER_API_KEY") or dotenv_values.get("OPENROUTER-KEY")
    if not api_key:
        raise RuntimeError("No OpenRouter key found (checked $OPENROUTER_API_KEY, and OPENROUTER_API_KEY/OPENROUTER-KEY in .env)")

    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("llm_runs", run_name)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(main.CUR_PHOTO_DIR, exist_ok=True)

    pg.init()
    screen = pg.display.set_mode((main.WIDTH, main.HEIGHT))
    smap = main.SatelliteMap()

    plane = Plane(main.START_X, main.START_Y, alt=alt, airspeed=airspeed, weight=weight, heading=heading, env_vars=env_vars, inst=None)

    trajectory = []
    turn = 0
    photo_path, size_nm, lat, lon = main.fetch_photo(smap, plane, turn, origin_lat, origin_lon)
    background = main.load_background(photo_path)

    while turn < max_turns:
        scale = main.render_frame(screen, plane, background, size_nm)
        pg.display.flip()
        frame_path = os.path.join(frames_dir, f"{turn:04d}.png")
        pg.image.save(screen, frame_path)

        if plane.landing:
            pg.image.save(screen, os.path.join(run_dir, "final.png"))
            break

        print(f"Turn {turn}: pos=({plane.pos_x:.2f},{plane.pos_y:.2f}) alt={plane.alt:.0f}ft -> asking LLM...")
        image_data_uri = frame_to_data_uri(screen)
        decision, raw_response, failed_attempts = get_llm_instruction(
            api_key, plane, size_nm, turn, max_turns, image_data_uri
        )
        print(f"  decision: {decision}")
        if failed_attempts:
            print(f"  ({len(failed_attempts)} failed attempt(s) before this decision)")

        goal_x = plane.pos_x + float(decision["rel_goal_x"])
        goal_y = plane.pos_y + float(decision["rel_goal_y"])
        annotate_frame_with_decision(frame_path, scale, goal_x, goal_y, decision)
        instruction = Instruction(
            goal_x=goal_x, goal_y=goal_y,
            airspeed=int(decision["airspeed"]), bank_angle=int(decision["bank_angle"]),
            flaps=int(decision["flaps"]),
        )

        trajectory.append({
            "turn": turn,
            "pos_x": plane.pos_x, "pos_y": plane.pos_y, "alt": plane.alt,
            "heading": plane.heading, "airspeed": plane.airspeed, "size_nm": size_nm,
            "lat": lat, "lon": lon, "frame": frame_path,
            "failed_attempts": failed_attempts,
            "llm_raw_response": raw_response,
            "instruction": {
                "goal_x": goal_x, "goal_y": goal_y, "airspeed": instruction.airspeed,
                "bank_angle": instruction.bank_angle, "flaps": instruction.flaps,
            },
        })

        plane.give_instruction(instruction)
        plane.follow_instruction()
        turn += 1
        photo_path, size_nm, lat, lon = main.fetch_photo(smap, plane, turn, origin_lat, origin_lon)
        background = main.load_background(photo_path)
    else:
        pg.image.save(screen, os.path.join(run_dir, "final.png"))

    summary = {
        "landed": bool(plane.landing),
        "turns": turn,
        "final_pos": [plane.pos_x, plane.pos_y],
        "final_alt": plane.alt,
        "aircraft_condition": plane.aircraft_condition,
        "model": OPENROUTER_MODEL,
    }
    with open(os.path.join(run_dir, "trajectory.json"), "w") as f:
        json.dump({"summary": summary, "turns": trajectory}, f, indent=2)

    pg.quit()
    print(f"Run complete: landed={summary['landed']} turns={turn} -> {run_dir}")
    return summary


if __name__ == "__main__":
    run(max_turns=20)
