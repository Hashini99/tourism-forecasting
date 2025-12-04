import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_monthly_weather(latitude, longitude, start_year, end_year):
    """
    Fetches daily weather from Open-Meteo and aggregates it to monthly data.
    Returns a DataFrame (year, month, mean_temp, max_temp, min_temp, total_precip).
    """
    all_months = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Calculate start and end date for the month
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            sdate = start_date.strftime("%Y-%m-%d")
            edate = end_date.strftime("%Y-%m-%d")
            
            # Build API URL
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={latitude}&longitude={longitude}"
                f"&start_date={sdate}&end_date={edate}"
                f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum"
                f"&timezone=Asia/Colombo"
            )
            
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"Error fetching {year}-{month}: {resp.status_code}")
                continue
            
            data = resp.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temps_mean = daily.get("temperature_2m_mean", [])
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            precs = daily.get("precipitation_sum", [])
            
            if not dates:
                continue
            
            # Create DataFrame for the month
            df = pd.DataFrame({
                "date": dates,
                "temp_mean": temps_mean,
                "temp_max": temps_max,
                "temp_min": temps_min,
                "precip_sum": precs
            })
            df["date"] = pd.to_datetime(df["date"])
            
            # Aggregate monthly
            monthly = {
                "year": year,
                "month": month,
                "mean_temp": df["temp_mean"].mean(),
                "max_temp": df["temp_max"].max(),
                "min_temp": df["temp_min"].min(),
                "total_precip": df["precip_sum"].sum()
            }
            all_months.append(monthly)
            
            # Sleep to respect API rate limits
            time.sleep(1)
    
    df_months = pd.DataFrame(all_months)
    return df_months.sort_values(["year", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    # Define main regions/stations
    stations = {
        "Colombo": (6.9271, 79.8612),
        "Kandy": (7.2906, 80.6337),
        "Galle": (6.0535, 80.2200),
        "Trincomalee": (8.5943, 81.2152),
        "Nuwara Eliya": (6.9690, 80.7800)
    }
    
    all_station_dfs = []
    
    # Fetch weather for all stations
    for city, (lat, lon) in stations.items():
        print(f"Fetching data for {city}...")
        df_station = fetch_monthly_weather(lat, lon, 2017, 2025)
        df_station["city"] = city
        all_station_dfs.append(df_station)
    
    # Combine all stations
    df_all = pd.concat(all_station_dfs)
    
    # Aggregate to country-level monthly data
    df_country = df_all.groupby(["year", "month"]).agg({
        "mean_temp": "mean",    # average of stations
        "max_temp": "max",      # hottest temperature
        "min_temp": "min",      # coldest temperature
        "total_precip": "sum"   # total rainfall
    }).reset_index()
    
    # Save to CSV
    df_country.to_csv("sri_lanka_monthly_weather_countrywide.csv", index=False)
    print("✅ Country-wide monthly weather saved as 'sri_lanka_monthly_weather_countrywide.csv'")
