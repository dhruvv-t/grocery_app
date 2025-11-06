
#!/usr/bin/env python3
# generate_sales_6m.py
# Efficient script to generate a large sales CSV (6M+ rows). Modify INPUT paths if needed.
import pandas as pd, numpy as np, uuid, math, os
from datetime import datetime

employees = pd.read_csv("../end_to_end/datasets/employees_expanded.csv")
customers = pd.read_csv("../end_to_end/datasets/customers_updated.csv")
products = pd.read_csv("../end_to_end/datasets/products_cleaned.csv")

product_ids = products['product_id'].values
product_prices = pd.to_numeric(products['price'], errors='coerce').fillna(0).values
prod_price_map = dict(zip(product_ids, product_prices))
prices_series = pd.Series(product_prices)
q25, q75 = prices_series.quantile(0.25), prices_series.quantile(0.75)

def sensible_qty_from_price(p):
    if p <= q25:
        return np.random.randint(1, 21)
    elif p <= q75:
        return np.random.randint(1, 11)
    else:
        return np.random.randint(1, 4)

total_rows_target = 6_000_000
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-12-31")
date_range = pd.date_range(start_date, end_date, freq="D")
num_days = len(date_range)
rows_per_day = math.ceil(total_rows_target / num_days)

# Prepare forced_by_day to guarantee inclusion
num_employees = employees.shape[0]
num_customers = customers.shape[0]
seed_days_employees = np.linspace(0, num_days-1, num=num_employees, dtype=int)
seed_days_customers = np.linspace(0, num_days-1, num=num_customers, dtype=int)
forced_by_day = {i: [] for i in range(num_days)}

for idx, row in employees.reset_index().iterrows():
    day_idx = int(seed_days_employees[idx])
    cust = int(np.random.choice(customers['customer_id'].values))
    prod = np.random.choice(product_ids)
    qty = sensible_qty_from_price(prod_price_map[prod])
    sec = np.random.randint(0, 24*60*60)
    forced_by_day[day_idx].append((int(row['employee_id']), cust, prod, int(qty), sec, str(uuid.uuid4())))

for idx, row in customers.reset_index().iterrows():
    day_idx = int(seed_days_customers[idx])
    emp = int(np.random.choice(employees['employee_id'].values))
    prod = np.random.choice(product_ids)
    qty = sensible_qty_from_price(prod_price_map[prod])
    sec = np.random.randint(0, 24*60*60)
    forced_by_day[day_idx].append((emp, int(row['customer_id']), prod, int(qty), sec, str(uuid.uuid4())))

out_path = "../end_to_end/datasets/sales_6m.csv"
if os.path.exists(out_path):
    os.remove(out_path)

header = "sales_id,employee_id,customer_id,product_id,quantity,sales_date,transaction_number\n"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(header)
    sales_id = 1
    rows_written = 0
    for day_idx, day in enumerate(date_range):
        forced = forced_by_day[day_idx]
        forced_count = len(forced)
        n = rows_per_day
        n_random = max(0, n - forced_count)
        emp_choices = np.random.choice(employees['employee_id'].values, size=n_random, replace=True)
        cust_choices = np.random.choice(customers['customer_id'].values, size=n_random, replace=True)
        prod_choices = np.random.choice(product_ids, size=n_random, replace=True)
        qty_choices = [sensible_qty_from_price(prod_price_map[p]) for p in prod_choices]
        times_random = np.random.randint(0, 24*60*60, size=n_random)
        txn_random = [str(uuid.uuid4()) for _ in range(n_random)]
        rows = []
        for emp, cust, prod, qty, sec, txn in forced:
            dt = (day + pd.Timedelta(seconds=int(sec))).strftime("%Y-%m-%d %H:%M:%S")
            line = f"{sales_id},{emp},{cust},{prod},{qty},{dt},{txn}\n"
            rows.append((sec, line)); sales_id += 1; rows_written += 1
        for emp, cust, prod, qty, sec, txn in zip(emp_choices, cust_choices, prod_choices, qty_choices, times_random, txn_random):
            dt = (day + pd.Timedelta(seconds=int(sec))).strftime("%Y-%m-%d %H:%M:%S")
            line = f"{sales_id},{int(emp)},{int(cust)},{prod},{int(qty)},{dt},{txn}\n"
            rows.append((int(sec), line)); sales_id += 1; rows_written += 1
        rows.sort(key=lambda x: x[0])
        f.writelines([r[1] for r in rows])
        if rows_written >= total_rows_target:
            break
print(f'Done. Wrote {rows_written} rows to {out_path}')
