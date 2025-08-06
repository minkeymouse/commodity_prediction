# import pandas as pd
#
# df = pd.read_csv('data/final_data/TRY_data.csv', parse_dates=['datetime'])
# df_usd = pd.read_csv('data/final_data/USD_data.csv', parse_dates=['datetime'])
#
# # Filter relevant data
# btc_try = df[df['market'] == 'BTCTurk'][['datetime', 'close']].rename(columns={'close': 'price_trybtc'})
# usd_try = df[df['market'] == 'USD_TRY'][['datetime', 'close']].rename(columns={'close': 'ex_usdtry'})
# usd_btc = df_usd[df_usd['market'] == '비트스탬프'][['datetime', 'close']].rename(columns={'close': 'price_usdbtc'})
#
# # Merge all on 'date'
# df_merged = pd.merge(btc_try, usd_try, on='datetime', how='inner')
# df_merged = pd.merge(df_merged, usd_btc, on='datetime', how='inner')
#
# print(df_merged)
# # Calculate Premium_TRY
# df_merged['premium_try'] = (
#     (df_merged['price_trybtc'] / df_merged['ex_usdtry'] - df_merged['price_usdbtc']) /
#     df_merged['price_usdbtc']
# )
#
# # Show result
# print(df_merged[['datetime', 'premium_try']])

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


# Currency-exchange mapping
currency_exchange_map = {
    "AUD": "BTC Markets", "BRL": "Mercado", "CAD": "크라켄", "CHF": "kraken", "GBP": "크라켄",
    "IDR": "Indodax", "INR": "Bitbns", "JPY": "비트플라이어", "KRW": ["Upbit", "빗썸", "코인원"], "MXN": "Bitso", "MYR": "Luno",
    "NGN": "Luno", "NZD": "Independent reserve", "PHP": "Coins_ph", "PLN": "Zonda", "THB": "Orbix", "TRY": "BTCTurk",
    "USD": "비트스탬프", "ZAR": "Luno"
}

# Load USD-based BTC price
df_usd = pd.read_csv('data/final_data/USD_data.csv', parse_dates=['datetime'])
usd_btc = df_usd[df_usd['market'] == '비트스탬프'][['datetime', 'close']].rename(columns={'close': 'price_usdbtc'})

# Container for result
premium_results = []

# Iterate through each currency
for ccy, exchange in currency_exchange_map.items():
    try:
        df = pd.read_csv(f'data/final_data/{ccy}_data.csv', parse_dates=['datetime'])
        df['close'] = pd.to_numeric(df['close'].astype(str).str.replace(',', '', regex=False), errors='coerce')  # ✅ safe
        if ccy == 'KRW':
            # Average price from multiple Korean exchanges
            ex_rate = df[df['market'] == 'USD_KRW'][['datetime', 'close']].rename(columns={'close': 'ex_usdccy'})
            print(df[df['market'] == 'USD_KRW'])
            print(ex_rate)
            krw_prices = df[df['market'].isin(["Upbit", "빗썸", "코인원"])]
            krw_avg = (
                krw_prices.groupby('datetime')['close']
                .mean()
                .reset_index()
                .rename(columns={'close': 'price_ccybtc'})
            )

            merged = pd.merge(krw_avg, ex_rate, on='datetime', how='inner')
            print(merged)

        else:
            # Regular case
            btc_ccy = df[df['market'] == exchange][['datetime', 'close']].rename(columns={'close': 'price_ccybtc'})
            ex_rate = df[df['market'] == f'USD_{ccy}'][['datetime', 'close']].rename(columns={'close': 'ex_usdccy'})
            merged = pd.merge(btc_ccy, ex_rate, on='datetime', how='inner')

        # Merge with USD/BTC price
        merged = pd.merge(merged, usd_btc, on='datetime', how='inner')

        # Calculate premium
        merged['premium'] = (
            (merged['price_ccybtc'] / merged['ex_usdccy'] - merged['price_usdbtc']) / merged['price_usdbtc']
        )

        merged['currency'] = ccy
        premium_results.append(merged[['datetime', 'currency', 'premium']])

    except Exception as e:
        print(f"Skipped {ccy} due to error: {e}")
        continue

# Combine all
df_premiums = pd.concat(premium_results, ignore_index=True)

# Show result
print(df_premiums)

df_wide = df_premiums.pivot(index='datetime', columns='currency', values='premium')
print(df_wide)

#

# # Plot for a few currencies
# sample = ['KRW', 'TRY', 'AUD', 'ZAR']
# df_sample = df_premiums[df_premiums['currency'].isin(sample)]
# for ccy in sample:
#     df_plot = df_sample[df_sample['currency'] == ccy]
#     plt.plot(df_plot['datetime'], df_plot['premium'], label=ccy)
#
# plt.legend()
# plt.title("BTC Premiums by Currency")
# plt.ylabel("Premium")
# plt.xlabel("Date")
# plt.grid(True)
# plt.show()

summary = df_premiums.groupby('currency')['premium'].agg(['mean', 'std', 'min', 'max']).sort_values('mean')
print(summary)