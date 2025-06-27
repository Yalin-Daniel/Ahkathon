import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime

API_BASE_URL = "http://localhost:8080"
app = dash.Dash(__name__)

# שליחת בקשת מחיקה לשרת - כדי שהמפה תתחיל נקייה בכל הרצה
try:
    res = requests.delete(f"{API_BASE_URL}/delete_all_videos/")
    if res.status_code == 200:
        print("✅ מסד הנתונים אופס בהצלחה")
    else:
        print(f"⚠️ שגיאה באיפוס הנתונים: {res.status_code}")
except Exception as e:
    print(f"⚠️ שגיאה בחיבור לשרת: {e}")

color_map = {
    "3+": "#ff4444",
    "1": "#44ff44",
    "2": "#4444ff",
}


def get_video_data_from_db():
    try:
        response = requests.get(f"{API_BASE_URL}/get_all_videos/")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"שגיאה בשליפת נתונים: {response.status_code}")
            return []
    except Exception as e:
        print(f"שגיאת חיבור למסד הנתונים: {e}")
        return []


def classify_activity_level(people_count):
    if people_count >= 3:
        return "3+"
    elif people_count == 2:
        return "2"
    elif people_count == 1:
        return "1"
    else:
        return None


def process_db_data_to_map_points():
    db_data = get_video_data_from_db()
    if not db_data:
        return pd.DataFrame(columns=[
            "lat", "lon", "type", "level", "size", "color",
            "people_count", "time", "frame_number", "video_path"
        ])

    points_data = []
    for entry in db_data:
        lat = entry.get('latitude', 0)
        lon = entry.get('longitude', 0)
        if lat == 0 and lon == 0:
            continue

        people_count = entry.get('pedestrian_count', 0)
        activity_type = classify_activity_level(people_count)

        point = {
            "lat": float(lat),
            "lon": float(lon),
            "type": activity_type,
            "level": min(5, max(1, people_count + 1)),
            "size": 2.0,  # קטן במיוחד
            "color": color_map.get(activity_type, "#888888"),
            "people_count": people_count,
            "time": entry.get('time', 'לא ידוע'),
            "frame_number": entry.get('frame_number', 0),
            "video_path": entry.get('video_path', 'לא ידוע')
        }
        points_data.append(point)

    return pd.DataFrame(points_data)


def create_map_figure(points_df, zoom=19):
    if points_df.empty:
        fig = px.scatter_mapbox(
            lat=[31.7855], lon=[35.19],
            zoom=zoom, height=700,
            mapbox_style="carto-positron"
        )
        fig.add_annotation(
            text="אין נתונים זמינים כרגע",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig

    fig = px.scatter_mapbox(
        points_df,
        lat="lat",
        lon="lon",
        color="type",
        hover_name="type",
        hover_data={
            "people_count": True,
            "time": True,
            "frame_number": True,
            "size": False,
            "lat": ":.6f",
            "lon": ":.6f"
        },
        zoom=zoom,
        height=700,
        mapbox_style="carto-positron",
        color_discrete_map=color_map,
        title="מפת זיהוי הולכי רגל בזמן אמת"
    )
    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        title={
            'text': "מפת זיהוי הולכי רגל בזמן אמת",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        font=dict(family="Arial", size=12),
        mapbox=dict(
            center=dict(
                lat=points_df['lat'].mean(),
                lon=points_df['lon'].mean()
            )
        )
    )
    return fig


app.layout = html.Div([
    html.Div([
        html.H1("מערכת מעקב הולכי רגל",
                style={'textAlign': 'center', 'color': '#2c3e50',
                       'marginBottom': '20px', 'fontFamily': 'Arial'}),

        html.Div([
            dcc.Graph(id="live-map"),

            html.Div([
                html.P(id="last-update",
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '10px'})
            ]),

            html.Div([
                html.Label("רמת תקריב (Zoom Level):"),
                dcc.Slider(id='zoom-slider', min=10, max=20, step=1, value=19,
                           marks={i: str(i) for i in range(10, 21)}, tooltip={"placement": "bottom"})
            ], style={'marginTop': '20px', 'padding': '10px 40px'})
        ])
    ], style={'padding': '20px'}),

    dcc.Interval(id='interval-component', interval=2 * 1000, n_intervals=0)
])


@app.callback(
    [Output('live-map', 'figure'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('zoom-slider', 'value')]
)
def update_map_live(n, zoom):
    points_df = process_db_data_to_map_points()
    fig = create_map_figure(points_df, zoom)
    last_update_text = f"עדכון אחרון: {datetime.now().strftime('%H:%M:%S')}"
    return fig, last_update_text


if __name__ == '__main__':
    print("🚀 מפעיל מפה דינמית...")
    print("📍 המפה תתעדכן כל 2 שניות")
    print("🌐 גש לכתובת: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)