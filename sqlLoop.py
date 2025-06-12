import time

from sql import get_hot_zones, get_latest_video_for_grid, process_grid_video, get_anomalies, \
    get_rapid_accumulation_zones, get_rapid_decrease_zones

SLEEP_TIME = 10  # המתנה בין סבבים (שניות)

def monitor_loop():
    while True:
        print("\n== Hot Zones ==")
        hot_zones = get_hot_zones()
        for grid_id, total_people in hot_zones:
            print(f"Grid {grid_id}: {total_people} people")
            video = get_latest_video_for_grid(grid_id)
            process_grid_video(grid_id, video)

        print("\n== Anomalies ==")
        anomalies = get_anomalies()
        for grid_id, people_count, avg_people, diff in anomalies:
            print(f"Grid {grid_id}: Current {people_count}, Average {avg_people:.2f}, Diff {diff}")
            video = get_latest_video_for_grid(grid_id)
            process_grid_video(grid_id, video)

        print("\n== Rapid Accumulation ==")
        rapid_accum = get_rapid_accumulation_zones()
        for row in rapid_accum:
            grid_id, prev_time, prev_count, latest_time, latest_count, growth_factor = row
            print(f"Grid {grid_id}: from {prev_count} to {latest_count}, growth x{growth_factor}")
            video = get_latest_video_for_grid(grid_id)
            process_grid_video(grid_id, video)

        print("\n== Rapid Decrease ==")
        rapid_decrease = get_rapid_decrease_zones()
        for row in rapid_decrease:
            grid_id, prev_time, prev_count, latest_time, latest_count, decrease_factor = row
            print(f"Grid {grid_id}: from {prev_count} to {latest_count}, decrease x{decrease_factor}")
            video = get_latest_video_for_grid(grid_id)
            process_grid_video(grid_id, video)

        print("\n----- Waiting for next check -----\n")
        time.sleep(SLEEP_TIME)