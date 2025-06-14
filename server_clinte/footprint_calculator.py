# footprint_calculator.py
import math
from geopy.distance import distance
from geopy import Point

def calculate_footprint(
        lat,
        lon,
        altitude_m,
        fov_horizontal_deg=90.0,
        fov_vertical_deg=72.0
):
    """
    מחשבת את תא השטח שהרחפן מצלם (בהנחה של צילום ישיר מטה)

    פרמטרים:
    - lat, lon: קואורדינטות של הרחפן
    - altitude_m: גובה מעל הקרקע
    - fov_horizontal_deg: זווית אופקית של המצלמה (ברירת מחדל 90°)
    - fov_vertical_deg: זווית אנכית של המצלמה (ברירת מחדל 72°)

    מחזירה:
    - מילון עם 4 פינות של תא השטח המצולם (NW, NE, SE, SW)
    """
    try:
        half_width = altitude_m
    except:
        pass