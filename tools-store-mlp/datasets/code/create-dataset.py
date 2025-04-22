import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


products = ['Hammer', 'Screwdriver', 'Pliers', 'Saw', 'Drill']
products_len = len(products)

initial_date = datetime(2022, 1, 1)

data_list = []
product_list = []
week_days_list = []
month_list = []
holiday_list = []
promotion_list = []
price_list = []
previous_sales_list = []
demand_list = []

for i in range(365):
    current_date = initial_date + timedelta(days=i)
    week_day = current_date.weekday()
    month = current_date.month
    holiday = 1 if current_date.day == 1 and month == 1 else 0 

    for product in products:
        product_list.append(product)
        data_list.append(current_date)
        week_days_list.append(week_day)
        month_list.append(month)
        holiday_list.append(holiday)
        promotion = 1 if random.random() < 0.15 else 0
        promotion_list.append(promotion)

        if product == 'Hammer':
            price = round(random.uniform(20, 40), 2)
        elif product == 'Screwdriver':
            price = round(random.uniform(10, 25), 2)
        elif product == 'Pliers':
            price = round(random.uniform(15, 35), 2)
        elif product == 'Saw':
            price = round(random.uniform(50, 120), 2)
        else: 
            price = round(random.uniform(80, 200), 2)
        price_list.append(price)
        previous_sales_list.append(0) #

        demand_base = 5 + 10 * (7 - abs(week_day - 3)) / 7 + 3 * (1 - abs(month - 6.5) / 5.5)
        if product == 'Drill':
            demand_base *= 1.2
        elif product == 'Hammer':
            demand_base *= 0.8

        demanda = max(0, int(demand_base + 15 * promotion + random.gauss(0, 5)))
        demand_list.append(demanda)

df = pd.DataFrame({
    'date': data_list,
    'product': product_list,
    'week_day': week_days_list,
    'month': month_list,
    'holiday': holiday_list,
    'promotion': promotion_list,
    'price': price_list,
    'previous_sales': previous_sales_list,
    'demand': demand_list
})

df['previous_sales'] = df.groupby('product')['demand'].shift(1).fillna(0).astype(int)

print(df.head())
print(f"\nSynthetic dataset of {len(df)} samples generated.")

df.to_csv('./../csv/tools.csv', index=False)
print("\nDataset saved in 'tools.csv'")