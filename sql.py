import sqlite3
import time
from datetime import datetime, timedelta

# ==================== קבועים ====================
COUNT_PEOPLE = 50
GRID_SIZE = 0.01
GROWTH_FACTOR = 2.0
DECREASE_FACTOR = 2.0
AVERAGE_DEVIATION = 20


# ==================== חיבור למסד ====================
conn = sqlite3.connect('videos.db')
cursor = conn.cursor()

# ==================== פונקציות עזר ====================

def init_db():
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        capture_time TEXT NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        people_count INTEGER NOT NULL,
        people_probability REAL NOT NULL,
        grid_id VARCHAR(50) NOT NULL
    )
    ''')
    conn.commit()
# פונקציה שמדפיסה את כל הסרטונים
def print_all_videos():
    cursor.execute('SELECT * FROM videos')
    rows = cursor.fetchall()
    for row in rows:
        print(row)

def compute_grid_id(lat, lon):
    lat_grid = round(lat / GRID_SIZE)
    lon_grid = round(lon / GRID_SIZE)
    return f"{lat_grid}_{lon_grid}"

def insert_video_entry(capture_time, lat, lon, people_count, people_probability):
    grid_id = compute_grid_id(lat, lon)
    cursor.execute("""
        INSERT INTO videos (capture_time, latitude, longitude, people_count, people_probability, grid_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (capture_time, lat, lon, people_count, people_probability, grid_id))
    conn.commit()

def get_latest_video_for_grid(grid_id):
    cursor.execute("""
        SELECT id, capture_time, latitude, longitude, people_count, people_probability
        FROM videos
        WHERE grid_id = ?
        ORDER BY capture_time DESC
        LIMIT 1
    """, (grid_id,))
    return cursor.fetchone()

def process_grid_video(grid_id, video_data):
    id, capture_time, latitude, longitude, people_count, people_probability = video_data
    print(f"\nGrid {grid_id}:")
    print(f"Time: {capture_time}")
    print(f"Location: ({latitude}, {longitude})")
    print(f"People: {people_count} (probability: {people_probability})")

# ==================== השאילתות ====================

def get_hot_zones():
    cursor.execute(f"""
        SELECT v.grid_id, SUM(v.people_count) AS total_people
        FROM videos v
        JOIN (
            SELECT grid_id, MAX(capture_time) AS last_time
            FROM videos
            GROUP BY grid_id
        ) latest
        ON v.grid_id = latest.grid_id AND v.capture_time = latest.last_time
        GROUP BY v.grid_id
        HAVING total_people > {COUNT_PEOPLE}
    """)
    return cursor.fetchall()

def get_anomalies():
    cursor.execute(f"""
        WITH averages AS (
            SELECT grid_id, AVG(people_count) AS avg_people
            FROM videos
            GROUP BY grid_id
        ),
        latest AS (
            SELECT v.grid_id, v.people_count, a.avg_people
            FROM videos v
            JOIN (
                SELECT grid_id, MAX(capture_time) AS last_time
                FROM videos
                GROUP BY grid_id
            ) latest_times
            ON v.grid_id = latest_times.grid_id AND v.capture_time = latest_times.last_time
            JOIN averages a
            ON v.grid_id = a.grid_id
        )
        SELECT grid_id, people_count, avg_people, (people_count - avg_people) AS diff
        FROM latest
        WHERE ABS(people_count - avg_people) > {AVERAGE_DEVIATION}
    """)
    return cursor.fetchall()

def get_rapid_accumulation_zones():
    cursor.execute(f"""
        WITH latest_per_grid AS (
            SELECT grid_id, MAX(capture_time) AS latest_time
            FROM videos
            GROUP BY grid_id
        ),
        latest_frames AS (
            SELECT v.* FROM videos v
            JOIN latest_per_grid lpg
            ON v.grid_id = lpg.grid_id AND v.capture_time = lpg.latest_time
        ),
        previous_frames AS (
            SELECT v.*, ROW_NUMBER() OVER (
                PARTITION BY v.grid_id ORDER BY datetime(v.capture_time) ASC
            ) as rn
            FROM videos v
            JOIN latest_per_grid lpg ON v.grid_id = lpg.grid_id
            WHERE datetime(v.capture_time) >= datetime(lpg.latest_time, '-10 minutes')
              AND datetime(v.capture_time) <= datetime(lpg.latest_time, '-5 minutes')
        ),
        earliest_prev AS (
            SELECT * FROM previous_frames WHERE rn = 1
        ),
        comparison AS (
            SELECT 
                l.grid_id, p.capture_time AS prev_time, p.people_count AS prev_count,
                l.capture_time AS latest_time, l.people_count AS latest_count,
                CASE WHEN p.people_count > 0 THEN CAST(l.people_count AS REAL) / p.people_count ELSE NULL END AS growth_ratio
            FROM latest_frames l JOIN earliest_prev p ON l.grid_id = p.grid_id
        )
        SELECT grid_id, prev_time, prev_count, latest_time, latest_count, ROUND(growth_ratio, 2) AS growth_factor
        FROM comparison
        WHERE growth_ratio >= {GROWTH_FACTOR}
        ORDER BY growth_ratio DESC
    """)
    return cursor.fetchall()

def get_rapid_decrease_zones():
    cursor.execute(f"""
        WITH latest_per_grid AS (
            SELECT grid_id, MAX(capture_time) AS latest_time
            FROM videos
            GROUP BY grid_id
        ),
        latest_frames AS (
            SELECT v.* FROM videos v
            JOIN latest_per_grid lpg
            ON v.grid_id = lpg.grid_id AND v.capture_time = lpg.latest_time
        ),
        previous_frames AS (
            SELECT v.*, ROW_NUMBER() OVER (
                PARTITION BY v.grid_id ORDER BY datetime(v.capture_time) ASC
            ) as rn
            FROM videos v
            JOIN latest_per_grid lpg ON v.grid_id = lpg.grid_id
            WHERE datetime(v.capture_time) >= datetime(lpg.latest_time, '-10 minutes')
              AND datetime(v.capture_time) <= datetime(lpg.latest_time, '-5 minutes')
        ),
        earliest_prev AS (
            SELECT * FROM previous_frames WHERE rn = 1
        ),
        comparison AS (
            SELECT 
                l.grid_id, p.capture_time AS prev_time, p.people_count AS prev_count,
                l.capture_time AS latest_time, l.people_count AS latest_count,
                CASE WHEN l.people_count > 0 THEN CAST(p.people_count AS REAL) / l.people_count ELSE NULL END AS decrease_ratio
            FROM latest_frames l JOIN earliest_prev p ON l.grid_id = p.grid_id
        )
        SELECT grid_id, prev_time, prev_count, latest_time, latest_count, ROUND(decrease_ratio, 2) AS decrease_factor
        FROM comparison
        WHERE decrease_ratio >= {DECREASE_FACTOR}
        ORDER BY decrease_ratio DESC
    """)
    return cursor.fetchall()