import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta
import time

# הגדרות מסד הנתונים
API_BASE_URL = "http://localhost:8080"

app = dash.Dash(__name__)

# מיפוי צבעים לפי סוג
color_map = {
    "decrease": "#ff4444",  # אדום
    "growth": "#44ff44",  # ירוק
    "average": "#4444ff",  # כחול
    "amount": "#ff44ff",  # סגול
    "high_activity": "#ffaa00",  # כתום
    "low_activity": "#888888"  # אפור
}


def get_video_data_from_db():
    """
    שליפת נתונים מהמסד נתונים
    """
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


def classify_activity_level(people_count, timestamp_diff_minutes=0):
    """
    סיווג רמת פעילות על בסיס כמות אנשים וזמן
    """
    if people_count >= 10:
        return "high_activity"
    elif people_count >= 5:
        return "growth"
    elif people_count >= 2:
        return "average"
    elif people_count == 1:
        return "low_activity"
    else:
        return "decrease"


def process_db_data_to_map_points():
    """
    עיבוד נתוני מסד הנתונים לנקודות מפה
    """
    db_data = get_video_data_from_db()

    if not db_data:
        return pd.DataFrame({
            "lat": pd.Series(dtype="float"),
            "lon": pd.Series(dtype="float"),
            "type": pd.Series(dtype="str"),
            "level": pd.Series(dtype="int"),
            "size": pd.Series(dtype="float"),
            "color": pd.Series(dtype="str"),
            "people_count": pd.Series(dtype="int"),
            "time": pd.Series(dtype="str"),
            "frame_number": pd.Series(dtype="int")
        })

    points_data = []
    points_without_location = 0

    for entry in db_data:
        lat = entry.get('latitude', 0)
        lon = entry.get('longitude', 0)

        # דילוג על נקודות ללא מיקום תקין
        if lat == 0 and lon == 0:
            points_without_location += 1
            continue

        people_count = entry.get('pedestrian_count', 0)
        activity_type = classify_activity_level(people_count)

        point = {
            "lat": float(lat),
            "lon": float(lon),
            "type": activity_type,
            "level": min(5, max(1, people_count + 1)),
            "size": float(min(50, max(10, people_count * 8 + 15))),
            "color": color_map.get(activity_type, "#888888"),
            "people_count": people_count,
            "time": entry.get('time', 'לא ידוע'),
            "frame_number": entry.get('frame_number', 0),
            "video_path": entry.get('video_path', 'לא ידוע')
        }
        points_data.append(point)

    if points_without_location > 0:
        print(f"⚠️ {points_without_location} נקודות ללא מיקום GPS נוספו")

    return pd.DataFrame(points_data)


def create_map_figure(points_df):
    """
    יצירת איור המפה
    """
    if points_df.empty:
        # מפה ריקה עם הודעה
        fig = px.scatter_mapbox(
            lat=[31.5], lon=[34.8],  # מרכז ישראל
            zoom=7, height=600,
            mapbox_style="open-street-map"
        )
        fig.add_annotation(
            text="אין נתונים זמינים כרגע",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig

    # יצירת המפה עם הנתונים
    fig = px.scatter_mapbox(
        points_df,
        lat="lat",
        lon="lon",
        color="type",
        size="size",
        hover_name="type",
        hover_data={
            "people_count": True,
            "time": True,
            "frame_number": True,
            "size": False,
            "lat": ":.6f",
            "lon": ":.6f"
        },
        zoom=7,
        height=600,
        mapbox_style="open-street-map",
        color_discrete_map=color_map,
        title="מפת זיהוי הולכי רגל בזמן אמת"
    )

    # עיצוב נוסף
    fig.update_layout(
        title={
            'text': "מפת זიהוי הולכי רגל בזמן אמת",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        font=dict(family="Arial", size=12),
        mapbox=dict(
            center=dict(
                lat=points_df['lat'].mean() if not points_df.empty else 31.5,
                lon=points_df['lon'].mean() if not points_df.empty else 34.8
            )
        )
    )

    return fig


# עיצוב האפליקציה
app.layout = html.Div([
    html.Div([
        html.H1("מערכת מעקב הולכי רגל",
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '20px',
                    'fontFamily': 'Arial'
                }),

        html.Div([
            html.Div(id="stats-container",
                     style={'marginBottom': '20px', 'textAlign': 'center'}),

            dcc.Graph(id="live-map"),

            html.Div([
                html.P(f"עדכון אחרון: {datetime.now().strftime('%H:%M:%S')}",
                       id="last-update",
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '10px'})
            ])
        ])
    ], style={'padding': '20px'}),

    # Interval component לעדכון אוטומטי
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # עדכון כל 5 שניות
        n_intervals=0
    )
])


@app.callback(
    [Output('live-map', 'figure'),
     Output('stats-container', 'children'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_map_live(n):
    """
    עדכון המפה בזמן אמת
    """
    try:
        # שליפת נתונים חדשים
        points_df = process_db_data_to_map_points()

        # יצירת המפה
        fig = create_map_figure(points_df)

        # חישוב סטטיסטיקות
        if not points_df.empty:
            total_people = points_df['people_count'].sum()
            total_locations = len(points_df)
            avg_people = points_df['people_count'].mean()
            max_people = points_df['people_count'].max()

            # קבלת סטטיסטיקות נוספות מהשרת
            try:
                response = requests.get(f"{API_BASE_URL}/get_video_stats/", timeout=5)
                if response.status_code == 200:
                    server_stats = response.json()
                    total_entries = server_stats.get('total_entries', 0)
                    entries_with_location = server_stats.get('entries_with_location', 0)
                    location_coverage = (entries_with_location / total_entries * 100) if total_entries > 0 else 0
                else:
                    total_entries = 0
                    entries_with_location = 0
                    location_coverage = 0
            except:
                total_entries = 0
                entries_with_location = 0
                location_coverage = 0

            stats = html.Div([
                html.Div([
                    html.H3(f"{total_people}", style={'margin': '0', 'color': '#3498db'}),
                    html.P("סה״כ אנשים", style={'margin': '0'})
                ], style={'display': 'inline-block', 'margin': '0 15px', 'textAlign': 'center'}),

                html.Div([
                    html.H3(f"{total_locations}", style={'margin': '0', 'color': '#2ecc71'}),
                    html.P("מיקומים עם GPS", style={'margin': '0'})
                ], style={'display': 'inline-block', 'margin': '0 15px', 'textAlign': 'center'}),

                html.Div([
                    html.H3(f"{avg_people:.1f}", style={'margin': '0', 'color': '#f39c12'}),
                    html.P("ממוצע אנשים", style={'margin': '0'})
                ], style={'display': 'inline-block', 'margin': '0 15px', 'textAlign': 'center'}),

                html.Div([
                    html.H3(f"{max_people}", style={'margin': '0', 'color': '#e74c3c'}),
                    html.P("מקסימום באתר", style={'margin': '0'})
                ], style={'display': 'inline-block', 'margin': '0 15px', 'textAlign': 'center'}),

                html.Div([
                    html.H3(f"{location_coverage:.1f}%", style={'margin': '0', 'color': '#9b59b6'}),
                    html.P("כיסוי GPS", style={'margin': '0'})
                ], style={'display': 'inline-block', 'margin': '0 15px', 'textAlign': 'center'})

            ], style={
                'backgroundColor': '#ecf0f1',
                'padding': '15px',
                'borderRadius': '10px',
                'marginBottom': '10px'
            })

            # הוספת מידע נוסף
            if total_entries > entries_with_location:
                missing_locations = total_entries - entries_with_location
                stats.children.append(
                    html.Div([
                        html.P(f"⚠️ {missing_locations} רשומות ללא מיקום GPS",
                               style={'textAlign': 'center', 'color': '#e67e22', 'margin': '10px 0 0 0'})
                    ])
                )
        else:
            stats = html.Div([
                html.H3("אין נתונים עם מיקום GPS", style={'color': '#e74c3c', 'textAlign': 'center'}),
                html.P("ודא שקובץ frames.xlsx קיים ומכיל נתוני מיקום תקינים",
                       style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '10px'})

        # זמן עדכון
        last_update_text = f"עדכון אחרון: {datetime.now().strftime('%H:%M:%S')}"

        return fig, stats, last_update_text

    except Exception as e:
        print(f"שגיאה בעדכון המפה: {e}")

        # החזרת מפה ריקה במקרה של שגיאה
        empty_fig = px.scatter_mapbox(
            lat=[31.5], lon=[34.8],
            zoom=7, height=600,
            mapbox_style="open-street-map"
        )
        empty_fig.add_annotation(
            text=f"שגיאה בטעינת נתונים: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )

        error_stats = html.Div([
            html.H3("שגיאה בטעינת נתונים", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"פרטים: {str(e)}", style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '10px'})

        return empty_fig, error_stats, f"שגיאה: {datetime.now().strftime('%H:%M:%S')}"


if __name__ == '__main__':
    print("🚀 מפעיל מפה דינמית...")
    print("📍 המפה תתעדכן כל 5 שניות")
    print("🌐 גש לכתובת: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)