# --- A-Realistic Grocery Sales Data Generator ---
# This script generates a large, synthetic sales dataset (6M+ rows)
# that simulates realistic, fluctuating demand patterns for
# different product categories.

import pandas as pd
import numpy as np
import uuid
import math
import os
from datetime import datetime, timedelta
from end_to_end.workflow.file_paths import paths

# Ensure the output directory exists
output_dir = os.path.dirname(paths['sales'])
if output_dir and not os.path.exists(output_dir):
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir)


# --- 2. Data Loading ---
# Re-using your function to read the source datasets
def read_datasets():
    """Reads the primary datasets needed for generation."""
    print("Reading source data (products, customers, employees)...")
    try:
        employees = pd.read_csv(paths['employees'])
        customers = pd.read_csv(paths['customers'])
        products = pd.read_csv(paths['products'])
        print("Source data loaded successfully.")
        return employees, customers, products
    except FileNotFoundError as e:
        print(f"Error: Missing source file. {e}")
        print("Please ensure 'products.csv', 'customers.csv', and 'employees.csv' exist at the paths defined in the 'paths' dictionary.")
        return None, None, None

# --- 3. Helper Functions (from your original script) ---
# Re-using your functions for quantity and price quartiles
def generate_product_quartile(products):
    """Calculates price quartiles and creates a product-price map."""
    product_ids = products['product_id'].values
    product_prices = pd.to_numeric(products['selling_price'], errors='coerce').fillna(0).values
    prod_price_map = dict(zip(product_ids, product_prices))
    prices_series = pd.Series(product_prices)
    q25, q75 = prices_series.quantile(0.25), prices_series.quantile(0.75)
    return q25, q75, product_ids, prod_price_map

def sensible_qty_from_price(p, q25, q75):
    """Generates a realistic quantity based on the product's price."""
    if p <= q25:
        # Cheaper items, higher quantity
        return np.random.randint(1, 11)
    elif p <= q75:
        # Mid-range items
        return np.random.randint(1, 6)
    else:
        # Expensive items, lower quantity
        return np.random.randint(1, 3)

# --- 4. NEW: Demand Profile Assignment ---
def assign_demand_profiles(products_df):
    """
    Assigns a demand profile to each product to simulate
    different sales patterns.
    """
    print("Assigning demand profiles to products...")
    product_to_profile = {}
    
    # Use quantile-based categorization on 'demand_score'
    # Ensure demand_score is numeric, fill NaNs with median
    if 'demand_score' not in products_df.columns:
        print("Warning: 'demand_score' column not found. Using random profiles.")
        products_df['demand_score'] = np.random.randint(1, 101, size=len(products_df))
        
    products_df['demand_score'] = pd.to_numeric(products_df['demand_score'], errors='coerce')
    products_df['demand_score'] = products_df['demand_score'].fillna(products_df['demand_score'].median())
    
    # Create 5 bins for demand profiles
    try:
        products_df['profile_bin'] = pd.qcut(products_df['demand_score'], 5, labels=[
            'lumpy', 'volatile', 'normal', 'stable', 'super_stable'
        ])
    except ValueError as e:
        print(f"Warning: Could not create bins with pd.qcut (likely due to non-unique quantiles). {e}")
        print("Falling back to random profile assignment.")
        products_df['profile_bin'] = np.random.choice(
            ['lumpy', 'volatile', 'normal', 'stable', 'super_stable'],
            size=len(products_df)
        )

    
    product_to_profile = dict(zip(products_df['product_id'], products_df['profile_bin']))
    
    # Override with 'seasonal' profile based on the 'seasonal_flag'
    if 'seasonal_flag' in products_df.columns:
        seasonal_products = products_df[products_df['seasonal_flag'] == True]['product_id']
        for pid in seasonal_products:
            product_to_profile[pid] = 'seasonal'
            
    # ---
    # Example of further customization (you can expand this):
    # Override for specific categories
    if 'category_name' in products_df.columns:
        trending_categories = ['Snacks & Confectionery', 'Beverages (Non-Alcoholic)']
        trending_products = products_df[products_df['category_name'].isin(trending_categories)]['product_id']
        
        # Assign 'trending' to 20% of these products
        for pid in trending_products.sample(frac=0.2):
             product_to_profile[pid] = 'trending'
    # ---

    print(f"Profiles assigned. Example: {list(product_to_profile.items())[:5]}")
    return product_to_profile

# --- 5. NEW: Daily Demand Score Simulation ---
def get_daily_demand_scores(current_date, product_to_profile, base_demand_map, event_state):
    """
    Calculates a demand score for every product based on its profile
    and the current date.
    """
    
    day_of_year = current_date.timetuple().tm_yday
    day_of_week = current_date.weekday() # 0=Monday, 6=Sunday
    
    # --- Global Modifiers ---
    # Higher sales on weekends
    weekend_boost = 1.6 if day_of_week >= 5 else 1.0
    
    # Global holiday seasonality (e.g., peak in December)
    # This is a sine wave peaking around day 350 (mid-December)
    holiday_effect = 1.0 + 0.4 * max(0, np.sin((day_of_year - 258) * (2 * np.pi) / 365))

    product_scores = {}
    new_event_state = event_state.copy()

    for pid, profile in product_to_profile.items():
        base = base_demand_map.get(pid, 50) # Get base demand, default to 50
        score = 0.0
        
        if profile == 'stable':
            # Consistent demand, low noise
            score = base + np.random.normal(0, 3)
        
        elif profile == 'super_stable':
            # Very consistent (e.g., milk, bread), very low noise
            score = base + np.random.normal(0, 1)

        elif profile == 'seasonal':
            # Strong summer peak (e.g., ice cream, drinks)
            # This sine wave peaks around day 172 (mid-June)
            seasonal_effect = 0.8 * np.sin((day_of_year - 80) * (2 * np.pi) / 365)
            score = base * (1 + seasonal_effect) + np.random.normal(0, 10)
            
        elif profile == 'volatile':
            # Simulates fads/events. Check if an event is active.
            if pid in event_state and event_state[pid] > 0:
                # Event is active
                score = base * 10 + np.random.normal(0, 20) # Huge spike
                new_event_state[pid] -= 1 # Countdown event days
            else:
                # No event, low-ish demand
                score = base * 0.5 + np.random.normal(0, 5)
                # 1% chance to start a new event
                if np.random.rand() < 0.01:
                    new_event_state[pid] = np.random.randint(3, 10) # Event lasts 3-10 days
                    
        elif profile == 'lumpy':
            # Sporadic sales. Most days, demand is near zero.
            if np.random.rand() < 0.1: # 10% chance of *any* demand
                score = base + np.random.normal(0, 5)
            else:
                score = 0.0 # 90% chance of zero demand
                
        elif profile == 'trending':
            # Linearly increasing demand over time
            days_since_start = (current_date - START_DATE).days
            growth_factor = 0.005 # 0.5% growth per day
            score = (base * (1 + growth_factor * days_since_start)) + np.random.normal(0, 8)

        else: # 'normal'
            # Base demand with standard noise
            score = base + np.random.normal(0, 10)

        # Apply global modifiers
        final_score = score * weekend_boost * holiday_effect
        
        # Ensure score is non-negative
        product_scores[pid] = max(0.01, final_score) # Use 0.01 to avoid zero-probability

    return product_scores, new_event_state


# --- 6. Main Data Generation Function (Modified) ---
def generate_sales_data():
    """
    Main function to generate the realistic sales data and
    write it to a CSV file.
    """
    
    # --- A. Setup ---
    global START_DATE # Make start date global for 'trending' calculation
    
    TARGET_ROWS = 12_000_000
    AVG_SALES_PER_DAY = 10_000 # 6M / 600 days
    N_DAYS = (TARGET_ROWS // AVG_SALES_PER_DAY) + 1 # ~601 days
    START_DATE = datetime(2023, 1, 2)

    print(f"Targeting {TARGET_ROWS} rows over approx. {N_DAYS} days.")
    
    employees, customers, products = read_datasets()
    if employees is None:
        return # Stop if data loading failed

    q25, q75, all_product_ids, prod_price_map = generate_product_quartile(products)
    
    employee_ids = employees['employee_id'].values
    customer_ids = customers['customer_id'].values

    # Create the product profile mappings
    product_to_profile = assign_demand_profiles(products)
    
    # Use 'demand_score' from products.csv as the base demand
    base_demand_map = dict(zip(products['product_id'], products['demand_score']))
    
    # State for 'volatile' items
    event_state = {} 

    # --- B. Generation Loop ---
    sales_id = 1
    rows_written = 0
    current_date = START_DATE
    
    output_filepath = paths['sales']
    
    print(f"Starting data generation. Writing to: {output_filepath}")

    try:
        with open(output_filepath, 'w') as output_file:
            # Write Header
            output_file.write("sales_id,employee_id,customer_id,product_id,quantity,sales_date,transaction_id\n")
            
            while rows_written < TARGET_ROWS:
                day_str = current_date.strftime("%Y-%m-%d")
                print(f"Generating data for {day_str}... ({rows_written}/{TARGET_ROWS} rows)", end='\r')

                # --- Step 1: Get Daily Demand ---
                daily_scores_map, event_state = get_daily_demand_scores(
                    current_date, 
                    product_to_profile, 
                    base_demand_map, 
                    event_state
                )
                
                # Convert map to an ordered list matching `all_product_ids`
                ordered_scores = [daily_scores_map.get(pid, 0.01) for pid in all_product_ids]
                
                # --- Step 2: Normalize to Probabilities ---
                total_score = sum(ordered_scores)
                if total_score == 0:
                    total_score = 1 # Avoid division by zero
                
                probabilities = np.array(ordered_scores) / total_score

                # --- Step 3: Determine Today's Sales Volume ---
                day_of_week_effect = 1.4 if current_date.weekday() >= 5 else 1.0
                n_sales_today = int(np.random.normal(AVG_SALES_PER_DAY * day_of_week_effect, 1000))
                
                if n_sales_today <= 0:
                    n_sales_today = 100 # Failsafe

                # --- Step 4: Generate Data (Vectorized) ---
                emp_choices = np.random.choice(employee_ids, size=n_sales_today, replace=True)
                cust_choices = np.random.choice(customer_ids, size=n_sales_today, replace=True)
                
                # *** THE CORE CHANGE ***
                # Use the weighted probabilities for product choice
                prod_choices = np.random.choice(
                    all_product_ids, 
                    size=n_sales_today, 
                    replace=True, 
                    p=probabilities
                )
                
                # Use list comprehension (fast enough for this)
                qty_choices = [sensible_qty_from_price(prod_price_map[p], q25, q75) for p in prod_choices]
                
                # Random times within the day
                times_random = np.random.randint(0, 24*60*60, size=n_sales_today)
                
                # Generate transaction IDs (chunked for realism)
                txn_ids = []
                avg_items_per_txn = 5
                n_txns = math.ceil(n_sales_today / avg_items_per_txn)
                for _ in range(n_txns):
                    txn_id = str(uuid.uuid4())
                    txn_ids.extend([txn_id] * avg_items_per_txn)
                
                # Shuffle and trim to size
                np.random.shuffle(txn_ids)
                txn_ids = txn_ids[:n_sales_today]

                # --- Step 5: Format and Write Rows ---
                rows_to_write = []
                for i in range(n_sales_today):
                    # Calculate final timestamp
                    sale_timestamp = (current_date + timedelta(seconds=int(times_random[i])))
                    dt_str = sale_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Create the line
                    line = f"{sales_id},{int(emp_choices[i])},{int(cust_choices[i])},{prod_choices[i]},{int(qty_choices[i])},{dt_str},{txn_ids[i]}\n"
                    
                    # Store as tuple (sort_key, line) for chronological sorting
                    rows_to_write.append((times_random[i], line))
                    sales_id += 1

                # Sort rows by time to be chronological
                rows_to_write.sort(key=lambda x: x[0])
                
                # Write all lines for the day
                output_file.writelines([row[1] for row in rows_to_write])
                
                # --- Step 6: Advance ---
                rows_written += n_sales_today
                current_date += timedelta(days=1)

    except IOError as e:
        print(f"\n\nError writing to file: {e}")
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")

    print(f"\nGeneration complete. Wrote {rows_written} rows to {output_filepath}.")

