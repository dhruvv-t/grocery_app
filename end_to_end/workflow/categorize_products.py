import pandas as pd

from end_to_end.workflow.file_paths import paths
from end_to_end.workflow import categorization_features


def load_dataset():
    
    sales = pd.read_csv(paths['sales_pro'])
    sales['sales_date'] = pd.to_datetime(sales['sales_date'])
    
    return sales



def categorize_all_products():
    sales = load_dataset()

    monthly_sales = categorization_features.calculate_monthly_sales(sales)

    feature_vector = categorization_features.calculate_std_dev(sales)
    feature_vector = categorization_features.calculate_coeff_of_variation(feature_vector)
    feature_vector = categorization_features.calculate_acf1(sales, feature_vector)
    feature_vector = categorization_features.calculate_seasonal_strength(monthly_sales, feature_vector)
    feature_vector = categorization_features.calculate_peakiness_and_spike_index(monthly_sales, feature_vector)
    feature_vector = categorization_features.calculate_zero_fraction(sales, feature_vector)
    feature_vector = categorization_features.calculate_trend_slope(monthly_sales, feature_vector)
    feature_vector = categorization_features.calculate_entropy(monthly_sales, feature_vector)

    feature_vector.to_csv(paths['cat_feature_vector'], index=False)

