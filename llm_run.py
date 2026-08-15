"""Automatically fly a scenario end-to-end using an OpenRouter vision LLM to choose each
instruction. Saves every annotated frame the LLM saw, the full decision trajectory (as JSON),
and a final frame showing the landing rollout path."""

import base64
import io
import json
import os
import time

import pygame as pg
import requests

import main
from plane import Plane, EnvironmentVariables, Instruction

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-4-31b-it"
MAX_TURNS = 20
MAX_PARSE_RETRIES = 3

SYSTEM_PROMPT = """You are the pilot of a Cessna 172 whose engine has failed at altitude. Glide the \
aircraft to the safest possible landing using the satellite image and flight data given each turn.

The image is centered on the plane (red dot with a red arrow showing its heading). Red gridlines \
are labeled with distance in nautical miles relative to the plane's CURRENT position: positive x \
is east, negative x is west, positive y is north, negative y is south. Prefer open, flat terrain \
(fields, roads, open ground) over water, forest, or buildings for the final approach.

Respond with ONLY a single JSON object (no markdown fences, no other text) with exactly these keys:
{"rel_goal_x": <float, nm east(+)/west(-) of current position>,
 "rel_goal_y": <float, nm north(+)/south(-) of current position>,
 "airspeed": <int, knots, typically 60-90>,
 "bank_angle": <int, degrees, typically 15-45>,
 "flaps": <int, one of 0, 10, 20, 30>,
 "forward_slip": <bool, true to shed extra altitude fast>,
 "reasoning": <short string explaining the choice>}
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


def build_user_prompt(plane, size_nm, turn):
    env = plane.environment_variables
    return (
        f"Turn {turn}. Altitude: {plane.alt:.0f} ft. Heading: {plane.heading:.0f} deg. "
        f"Airspeed: {plane.airspeed} kt. Wind: {env.wind_direction} deg @ {env.wind_strength} kt. "
        f"Temperature: {env.temperature}C. The image spans {size_nm:.2f} nm across, centered on "
        f"the plane. Choose the next instruction."
    )


def frame_to_data_uri(surface):
    buf = io.BytesIO()
    pg.image.save(surface, buf, "frame.png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_openrouter(api_key, user_text, image_data_uri):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]},
        ],
        "temperature": 0.4,
    }
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
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    data = json.loads(text)
    for key in ["rel_goal_x", "rel_goal_y", "airspeed", "bank_angle", "flaps", "forward_slip"]:
        if key not in data:
            raise ValueError(f"missing key: {key}")
    return data


def sanitize_decision(decision):
    """Clamp/snap LLM output to values plane.py's physics can actually handle: bank_angle=0
    causes a division by zero in the turn-radius formula, airspeed=0 similarly breaks the
    glide-ratio formula, and flaps must be exactly one of Instruction's allowed values."""
    decision = dict(decision)
    decision["bank_angle"] = max(5, min(60, int(decision["bank_angle"])))
    decision["airspeed"] = max(40, min(120, int(decision["airspeed"])))
    decision["flaps"] = min([0, 10, 20, 30], key=lambda f: abs(f - int(decision["flaps"])))
    return decision


def get_llm_instruction(api_key, plane, size_nm, turn, image_data_uri):
    user_text = build_user_prompt(plane, size_nm, turn)
    last_error = None
    for attempt in range(MAX_PARSE_RETRIES):
        raw = call_openrouter(api_key, user_text, image_data_uri)
        try:
            return parse_instruction_json(raw), raw
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"  (parse failed, attempt {attempt + 1}/{MAX_PARSE_RETRIES}: {e})")
    raise RuntimeError(f"Could not get a valid instruction after {MAX_PARSE_RETRIES} attempts: {last_error}")


def run(run_name=None, max_turns=MAX_TURNS):
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

    env = EnvironmentVariables(wind_strength=8, wind_direction=270, temperature=15)
    plane = Plane(main.START_X, main.START_Y, alt=1500, airspeed=80, weight=2400, heading=270, env_vars=env, inst=None)

    trajectory = []
    turn = 0
    glide_ratio = main.scenario_glide_ratio(plane)
    photo_path, size_nm, lat, lon = main.fetch_photo(smap, plane, turn, glide_ratio)
    background = main.load_background(photo_path)

    while turn < max_turns:
        main.render_frame(screen, plane, background, size_nm)
        pg.display.flip()
        frame_path = os.path.join(frames_dir, f"{turn:04d}.png")
        pg.image.save(screen, frame_path)

        if plane.landing:
            pg.image.save(screen, os.path.join(run_dir, "final.png"))
            break

        print(f"Turn {turn}: pos=({plane.pos_x:.2f},{plane.pos_y:.2f}) alt={plane.alt:.0f}ft -> asking LLM...")
        image_data_uri = frame_to_data_uri(screen)
        decision, raw_response = get_llm_instruction(api_key, plane, size_nm, turn, image_data_uri)
        decision = sanitize_decision(decision)
        print(f"  decision: {decision}")

        goal_x = plane.pos_x + float(decision["rel_goal_x"])
        goal_y = plane.pos_y + float(decision["rel_goal_y"])
        instruction = Instruction(
            goal_x=goal_x, goal_y=goal_y,
            airspeed=int(decision["airspeed"]), bank_angle=int(decision["bank_angle"]),
            flaps=int(decision["flaps"]), forward_slip=bool(decision["forward_slip"]),
        )

        trajectory.append({
            "turn": turn,
            "pos_x": plane.pos_x, "pos_y": plane.pos_y, "alt": plane.alt,
            "heading": plane.heading, "airspeed": plane.airspeed, "size_nm": size_nm,
            "lat": lat, "lon": lon, "frame": frame_path,
            "llm_raw_response": raw_response, "llm_decision": decision,
            "instruction": {
                "goal_x": goal_x, "goal_y": goal_y, "airspeed": instruction.airspeed,
                "bank_angle": instruction.bank_angle, "flaps": instruction.flaps,
                "forward_slip": instruction.forward_slip,
            },
        })

        plane.give_instruction(instruction)
        plane.follow_instruction()
        turn += 1
        photo_path, size_nm, lat, lon = main.fetch_photo(smap, plane, turn, glide_ratio)
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
    run()
