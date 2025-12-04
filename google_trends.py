# google_trends.py

from pytrends.request import TrendReq
import pandas as pd

# 1️⃣ Connect to Google
pytrends = TrendReq(hl='en-US', tz=330)  # tz=330 for Sri Lanka time

# 2️⃣ Define keywords and timeframe
keywords = ["Sri Lanka tourism", "Sri Lanka travel"]
timeframe = '2017-01-01 2024-12-31'

# 3️⃣ Build the payload
pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo='')

# 4️⃣ Fetch interest over time
data = pytrends.interest_over_time()

if data.empty:
    print("No data returned. Check your keywords or timeframe.")
else:
    # Remove 'isPartial' column if exists
    if 'isPartial' in data.columns:
        data = data.drop(columns=['isPartial'])

    # Convert index to datetime
    data.index = pd.to_datetime(data.index)

    # Resample to monthly average
    monthly_data = data.resample('M').mean()

    # Reset index for CSV
    monthly_data.reset_index(inplace=True)
    monthly_data.rename(columns={'date':'year_month'}, inplace=True)

    # Save as CSV
    monthly_data.to_csv("google_trends_srilanka.csv", index=False)
    print("Saved monthly Google Trends data to google_trends_srilanka.csv")
