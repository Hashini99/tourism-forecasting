import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_monthly_weather(latitude, longitude, start_year, end_year, output_csv):
    """
    Fetches daily data and aggregates to monthly from Open-Meteo, then writes CSV.
    """
    all_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Build start_date and end_date for that month
            start_date = datetime(year, month, 1)
            # Next month, minus one day
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            sdate = start_date.strftime("%Y-%m-%d")
            edate = end_date.strftime("%Y-%m-%d")
            
            # request daily data
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={latitude}&longitude={longitude}"
                f"&start_date={sdate}&end_date={edate}"
                f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum"
                f"&timezone=Asia/Colombo"
            )
            
            resp = requests.get(url)
            if resp.status_code != 200:
                print("Error:", resp.status_code, resp.text)
                # You may want to retry or skip
                continue
            
            data = resp.json()
            # daily arrays
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temps_mean = daily.get("temperature_2m_mean", [])
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            precs = daily.get("precipitation_sum", [])
            
            # create DataFrame
            df = pd.DataFrame({
                "date": dates,
                "temp_mean": temps_mean,
                "temp_max": temps_max,
                "temp_min": temps_min,
                "precip_sum": precs
            })
            if df.empty:
                continue
            
            # convert date to datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # aggregate monthly
            monthly = {
                "year": year,
                "month": month,
                "mean_temp": df["temp_mean"].mean(),
                "max_temp": df["temp_max"].max(),
                "min_temp": df["temp_min"].min(),
                "total_precip": df["precip_sum"].sum()
            }
            all_months.append(monthly)
            
            # To respect rate limits
            time.sleep(1)
    
    # Convert to DataFrame
    df_months = pd.DataFrame(all_months)
    # Sort by year, month
    df_months = df_months.sort_values(["year", "month"])
    # Save
    df_months.to_csv(output_csv, index=False)
    print("Saved to", output_csv)


if __name__ == "__main__":
    # Example for Colombo, Sri Lanka
    lat = 6.9271
    lon = 79.8612
    fetch_monthly_weather(lat, lon, start_year=2017, end_year=2025, output_csv="sri_lanka_monthly_weather.csv")
