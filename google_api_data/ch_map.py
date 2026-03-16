import os
import requests
import folium
import polyline as polyline_decoder
import time
import json
from datetime import datetime, timezone


API_KEY = "AIzaSyD13u47SHc5WQXPqxH_F2q1Tlb4pnBx1bU"
OUTPUT_HTML = "chandigarh_traffic_map.html"
OUTPUT_JSON = "chandigarh_traffic_data.json"
SNAPSHOT_DIR = "traffic_snapshots"
ITERATIONS = 10
INTERVAL_SECONDS = 60 * 60  # hourly

# CHANDIGARH ROAD SEGMENTS
# Each tuple: (name, (start_lat, start_lng), (end_lat, end_lng))
ROAD_SEGMENTS = [
    # MAJOR ARTERIALS
    ("Madhya Marg (N-S Spine)", (30.7333, 76.7794), (30.6934, 76.7794)),
    ("Jan Marg", (30.7280, 76.7650), (30.6980, 76.7650)),
    ("Udyog Path", (30.7200, 76.7550), (30.6900, 76.7550)),
    ("Himalaya Marg", (30.7400, 76.7900), (30.7100, 76.7900)),
    ("Dakshin Marg", (30.7100, 76.8050), (30.6800, 76.8050)),
    ("Purv Marg", (30.7200, 76.8200), (30.6900, 76.8200)),
    ("Uttar Marg", (30.7500, 76.7700), (30.7500, 76.8100)),

    # SECTOR DIVIDERS (E-W)
    ("Sector 7-8 Road", (30.7480, 76.7650), (30.7480, 76.8100)),
    ("Sector 8-9 Road", (30.7430, 76.7650), (30.7430, 76.8100)),
    ("Sector 9-10 Road", (30.7380, 76.7650), (30.7380, 76.8100)),
    ("Sector 10-11 Road", (30.7330, 76.7650), (30.7330, 76.8100)),
    ("Sector 14-15 Road", (30.7280, 76.7650), (30.7280, 76.8100)),
    ("Sector 15-16 Road", (30.7230, 76.7650), (30.7230, 76.8100)),
    ("Sector 16-17 Road", (30.7180, 76.7650), (30.7180, 76.8100)),
    ("Sector 17-18 Road", (30.7130, 76.7650), (30.7130, 76.8100)),
    ("Sector 20-21 Road", (30.7080, 76.7650), (30.7080, 76.8100)),
    ("Sector 21-22 Road", (30.7030, 76.7650), (30.7030, 76.8100)),
    ("Sector 22-23 Road", (30.6980, 76.7650), (30.6980, 76.8100)),
    ("Sector 34-35 Road", (30.7200, 76.7550), (30.7200, 76.7650)),
    ("Sector 35-36 Road", (30.7150, 76.7550), (30.7150, 76.7650)),
    ("Sector 36-37 Road", (30.7100, 76.7550), (30.7100, 76.7650)),

    # KEY INTERSECTIONS / CONNECTOR ROADS
    ("Airport Road", (30.6720, 76.7880), (30.6500, 76.7700)),
    ("Chandigarh-Panchkula Link", (30.7500, 76.8100), (30.7500, 76.8500)),
    ("Chandigarh-Mohali Link", (30.7100, 76.7600), (30.7100, 76.7200)),
    ("IT Park Road", (30.7200, 76.7300), (30.7400, 76.7300)),
    ("Sector 43 Market Road", (30.7050, 76.7900), (30.7050, 76.8100)),
    ("ISBT 43 Connector", (30.7000, 76.8050), (30.7100, 76.8050)),
    ("PGI Hospital Road", (30.7650, 76.7780), (30.7500, 76.7780)),
    ("Sector 32 Hospital Road", (30.7220, 76.7800), (30.7220, 76.7950)),
    ("Rock Garden Road", (30.7550, 76.8080), (30.7650, 76.8050)),
    ("Sukhna Lake Road", (30.7430, 76.8200), (30.7550, 76.8100)),
    ("Rose Garden Road", (30.7340, 76.7850), (30.7340, 76.8050)),
    ("Sector 17 Plaza Loop", (30.7400, 76.7800), (30.7350, 76.7850)),
    ("Tribune Chowk Connector", (30.6880, 76.7750), (30.7000, 76.7750)),
    ("Hallomajra Link", (30.7100, 76.8300), (30.7100, 76.8500)),
    ("Manimajra Connector", (30.7050, 76.8600), (30.7200, 76.8600)),
    ("Burail Road", (30.7600, 76.7600), (30.7600, 76.7900)),
    ("Dhanas Road", (30.7700, 76.7500), (30.7700, 76.7700)),
    ("Sector 48-49 Road", (30.6900, 76.7700), (30.6900, 76.8000)),
    ("VIP Road (Sector 28-29)", (30.7170, 76.8000), (30.7170, 76.8200)),
    ("Sector 44-45 Connector", (30.6980, 76.7800), (30.7080, 76.7800)),

    # PANCHKULA EXTENSION
    ("Panchkula Sector 5 Road", (30.6950, 76.8550), (30.7100, 76.8550)),
    ("Panchkula Sector 10-11 Road", (30.7100, 76.8400), (30.7200, 76.8400)),
    ("Panchkula MDR-132", (30.6900, 76.8300), (30.7000, 76.8450)),

    # MOHALI EXTENSION
    ("Mohali Phase 7 Road", (30.7150, 76.7100), (30.7000, 76.7100)),
    ("Mohali Phase 8 Road", (30.7300, 76.7050), (30.7150, 76.7050)),
    ("Aerocity Road", (30.6700, 76.7500), (30.6850, 76.7500)),
    ("Sector 66 Mohali", (30.6950, 76.7200), (30.7100, 76.7200)),
    ("Mohali IT City Road", (30.7400, 76.7000), (30.7250, 76.7000)),
    ("Landran Road", (30.7100, 76.6900), (30.7300, 76.6900)),
    ("Kharar Link Road", (30.7500, 76.6800), (30.7400, 76.7000)),

    # INNER SECTOR ROADS
    ("Sector 17 Inner Loop", (30.7390, 76.7820), (30.7340, 76.7870)),
    ("Sector 22 Market Road", (30.7330, 76.7780), (30.7290, 76.7820)),
    ("Sector 34 Market Road", (30.7230, 76.7700), (30.7190, 76.7740)),
    ("Sector 35 Market Road", (30.7240, 76.7640), (30.7200, 76.7680)),
    ("Sector 8 Inner Road", (30.7440, 76.7780), (30.7480, 76.7840)),
    ("Sector 11 Inner Road", (30.7370, 76.7860), (30.7330, 76.7900)),
    ("Sector 15 Inner Road", (30.7280, 76.7700), (30.7240, 76.7740)),
    ("Sector 20 Market", (30.7100, 76.7900), (30.7060, 76.7940)),
    ("Sector 37 Market", (30.7060, 76.7700), (30.7020, 76.7740)),
    ("Sector 40 Inner", (30.7030, 76.7820), (30.6990, 76.7860)),
    ("Sector 43 Inner", (30.7050, 76.7960), (30.7010, 76.8000)),
    ("Sector 44 Inner", (30.7000, 76.7900), (30.6960, 76.7940)),
    ("Sector 46 Inner", (30.6960, 76.8000), (30.6920, 76.8040)),
    ("Sector 47 Inner", (30.6950, 76.7850), (30.6910, 76.7890)),

    # OUTER RING / BYPASS
    ("NH-5 (Chandigarh-Ambala)", (30.7800, 76.8000), (30.7650, 76.7900)),
    ("NH-7 (Chandigarh-Shimla)", (30.7600, 76.8300), (30.7700, 76.8500)),
    ("NH-21 (Chandigarh-Ropar)", (30.7500, 76.8500), (30.7600, 76.8700)),
    ("Zirakpur Bypass", (30.6500, 76.8100), (30.6700, 76.8300)),
    ("Kalka Road", (30.7600, 76.8400), (30.7700, 76.8600)),
    ("Chandigarh Ring Road (N)", (30.7700, 76.7700), (30.7700, 76.8200)),
    ("Chandigarh Ring Road (S)", (30.6800, 76.7700), (30.6800, 76.8200)),
    ("Chandigarh Ring Road (W)", (30.7700, 76.7700), (30.6800, 76.7700)),
    ("Chandigarh Ring Road (E)", (30.7700, 76.8200), (30.6800, 76.8200)),
]

SPEED_COLORS = {
    "NORMAL": "#22c55e",      # green
    "SLOW": "#f97316",        # orange
    "TRAFFIC_JAM": "#ef4444", # red
    "UNKNOWN": "#94a3b8",     # grey (fallback)
}

SPEED_LABELS = {
    "NORMAL": "Normal Flow",
    "SLOW": "Slow Traffic",
    "TRAFFIC_JAM": "Traffic Jam",
    "UNKNOWN": "No Data",
}

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


def get_traffic_for_segment(name, origin, destination):
    """
    Calls the Google Routes API for a single road segment.
    Returns list of (decoded_polyline_points, speed_category) tuples.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.polyline,routes.legs.polyline,routes.travelAdvisory.speedReadingIntervals",
    }

    body = {
        "origin": {
            "location": {"latLng": {"latitude": origin[0], "longitude": origin[1]}}
        },
        "destination": {
            "location": {"latLng": {"latitude": destination[0], "longitude": destination[1]}}
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "polylineQuality": "HIGH_QUALITY",
        "extraComputations": ["TRAFFIC_ON_POLYLINE"],
    }

    try:
        resp = requests.post(ROUTES_API_URL, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "routes" not in data or not data["routes"]:
            print(f"  No route returned for: {name}")
            return None

        route = data["routes"][0]
        encoded = route["polyline"]["encodedPolyline"]
        points = polyline_decoder.decode(encoded)

        # Extract speed intervals
        intervals = []
        try:
            intervals = route["travelAdvisory"]["speedReadingIntervals"]
        except KeyError:
            pass

        if not intervals:
            return [(points, "NORMAL")]

        # Build list of (segment_points, speed)
        segments = []
        for interval in intervals:
            start_idx = interval.get("startPolylinePointIndex", 0)
            end_idx = interval.get("endPolylinePointIndex", len(points) - 1)
            speed = interval.get("speed", "UNKNOWN")
            seg_points = points[start_idx:end_idx + 1]
            if len(seg_points) >= 2:
                segments.append((seg_points, speed))

        return segments if segments else [(points, "NORMAL")]

    except requests.exceptions.HTTPError as e:
        print(f"  HTTP error for {name}: {e}")
        response = e.response
        if response is not None:
            try:
                error_obj = response.json().get("error", {})
                error_message = error_obj.get("message", "")
                details = error_obj.get("details", [])
                activation_url = None
                reason = None
                for detail in details:
                    if detail.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo":
                        reason = detail.get("reason")
                        activation_url = detail.get("metadata", {}).get("activationUrl")
                        break
                if error_message:
                    print(f"    API message: {error_message}")
                if reason:
                    print(f"    Reason: {reason}")
                if activation_url:
                    print(f"    Enable here: {activation_url}")
            except Exception:
                pass
        return None
    except Exception as e:
        print(f"  Error for {name}: {e}")
        return None


def build_map(results):
    """Renders all traffic segments onto a Folium map."""
    m = folium.Map(
        location=[30.7333, 76.7794],  # Chandigarh center
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )

    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(15,15,20,0.92); padding: 14px 18px;
        border-radius: 10px; border: 1px solid #333;
        font-family: monospace; font-size: 13px; color: #eee;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <b style="font-size:14px;">Chandigarh Traffic</b><br>
        <span style="color:#94a3b8;">Generated: {ts}</span><br><br>
        <span style="color:#22c55e;">■</span> Normal Flow<br>
        <span style="color:#f97316;">■</span> Slow Traffic<br>
        <span style="color:#ef4444;">■</span> Traffic Jam<br>
        <span style="color:#94a3b8;">■</span> No Data
    </div>
    """.format(ts=datetime.now().strftime("%d %b %Y, %I:%M %p"))
    m.get_root().html.add_child(folium.Element(legend_html))

    total_segments = 0
    stats = {"NORMAL": 0, "SLOW": 0, "TRAFFIC_JAM": 0, "UNKNOWN": 0}

    for road_name, segment_list in results:
        if segment_list is None:
            continue
        for (pts, speed) in segment_list:
            if len(pts) < 2:
                continue
            color = SPEED_COLORS.get(speed, SPEED_COLORS["UNKNOWN"])
            folium.PolyLine(
                locations=pts,
                color=color,
                weight=5,
                opacity=0.85,
                tooltip=f"{road_name} - {SPEED_LABELS.get(speed, speed)}",
            ).add_to(m)
            stats[speed] = stats.get(speed, 0) + 1
            total_segments += 1

    print(f"\nRendered {total_segments} polyline segments")
    print(f"   Normal:      {stats['NORMAL']}")
    print(f"   Slow:        {stats['SLOW']}")
    print(f"   Traffic Jam: {stats['TRAFFIC_JAM']}")
    print(f"   No Data:     {stats['UNKNOWN']}")

    return m


def _timestamp_slug(dt_utc):
    return dt_utc.strftime("%Y%m%d_%H%M%SZ")


def collect_and_store_snapshot(iteration, total_iterations):
    iteration_started = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"Iteration {iteration}/{total_iterations}")
    print(f"Started at (UTC): {iteration_started.isoformat()}")
    print("=" * 60)

    results = []
    raw_data = []

    for i, (name, origin, dest) in enumerate(ROAD_SEGMENTS, 1):
        print(f"[{i:>3}/{len(ROAD_SEGMENTS)}] {name}")
        segments = get_traffic_for_segment(name, origin, dest)
        results.append((name, segments))

        # Store raw data for JSON export
        raw_data.append({
            "road": name,
            "origin": origin,
            "destination": dest,
            "segments": [
                {"speed": spd, "points": pts}
                for (pts, spd) in (segments if segments else [])
            ]
        })

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    print("\nBuilding map...")
    m = build_map(results)

    ts_slug = _timestamp_slug(iteration_started)
    snapshot_html = os.path.join(SNAPSHOT_DIR, f"chandigarh_traffic_map_{ts_slug}.html")
    snapshot_json = os.path.join(SNAPSHOT_DIR, f"chandigarh_traffic_data_{ts_slug}.json")

    m.save(snapshot_html)
    m.save(OUTPUT_HTML)  # keep a latest copy for convenience
    print(f"Map snapshot saved -> {snapshot_html}")
    print(f"Latest map saved   -> {OUTPUT_HTML}")

    payload = {
        "generated_at": iteration_started.isoformat(),
        "iteration": iteration,
        "total_iterations": total_iterations,
        "total_roads": len(ROAD_SEGMENTS),
        "roads": raw_data,
    }

    with open(snapshot_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Raw snapshot saved -> {snapshot_json}")
    print(f"Latest raw saved   -> {OUTPUT_JSON}")
    print(f"Open {snapshot_html} in your browser to view this snapshot.")


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your API key in the script (API_KEY variable)")
        return

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("=" * 60)
    print("Chandigarh Traffic Map - Hourly Snapshot Collector")
    print("=" * 60)
    print(f"Road segments to query: {len(ROAD_SEGMENTS)}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Interval: {INTERVAL_SECONDS} seconds (hourly)")
    print(f"Snapshot directory: {SNAPSHOT_DIR}")
    print("=" * 60)

    for iteration in range(1, ITERATIONS + 1):
        cycle_start = time.time()
        collect_and_store_snapshot(iteration, ITERATIONS)

        if iteration < ITERATIONS:
            elapsed = time.time() - cycle_start
            wait_seconds = max(0, INTERVAL_SECONDS - int(elapsed))
            wait_minutes = round(wait_seconds / 60, 2)
            print(f"\nWaiting {wait_seconds} seconds (~{wait_minutes} min) before next run...\n")
            time.sleep(wait_seconds)

    print("\nAll hourly iterations completed.")


if __name__ == "__main__":
    main()
