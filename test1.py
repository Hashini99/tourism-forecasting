import requests
import json

weather_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 6.9271,
    "longitude": 79.8612,
    "start_date": "2017-01-01",
    "end_date": "2025-12-31",
    "monthly": "temperature_2m_mean,precipitation_sum",
    "timezone": "auto"
}

response = requests.get(weather_url, params=params)
weather_json = response.json()

print("Top-level keys in weather_json:", list(weather_json.keys()))
