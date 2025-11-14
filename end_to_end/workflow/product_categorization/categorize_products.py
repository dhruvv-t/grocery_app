import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from end_to_end.workflow.file_paths import paths
from end_to_end.workflow.product_categorization import categorization_features


# Utility Functions
def standardize_features(arrX, feature_vector):

    for i in arrX:
        feature_vector[i] = (
            (feature_vector[i] - feature_vector.describe()[i]['min']) / (feature_vector.describe()[i]['max'] - feature_vector.describe()[i]['min'])
        ).round(3)

    return feature_vector


# Mainstream functions
def save_product_wise_feature_vector():
    
    sales = pd.read_csv(paths['sales_pro'])
    sales['sales_date'] = pd.to_datetime(sales['sales_date'])

    monthly_sales = categorization_features.calculate_monthly_sales(sales)

    feature_vector = categorization_features.calculate_std_dev(sales)
    feature_vector = categorization_features.calculate_coeff_of_variation(feature_vector)
    feature_vector = categorization_features.calculate_acf1(sales, feature_vector)
    feature_vector = categorization_features.calculate_seasonal_strength(monthly_sales, feature_vector)
    feature_vector = categorization_features.calculate_peakiness_and_spike_index(monthly_sales, feature_vector)
    feature_vector = categorization_features.calculate_zero_fraction(sales, feature_vector)
    feature_vector = categorization_features.calculate_trend_slope(monthly_sales, feature_vector)
    feature_vector = categorization_features.calculate_entropy(monthly_sales, feature_vector)
    print("Calculated all the features...                ")
    print("Saving the Feature Vector as a CSV file")

    feature_vector.to_csv(paths['cat_feature_vector'], index=False)

    return feature_vector
    

def categorize_products_on_sales():
    
    print('Starting Product Level categorization.')
    feature_vector = save_product_wise_feature_vector()
    arr_features_to_normalize = ['coeff_variation', 'autocorr_lag1', 'seasonal_strength', 'entropy']

    print('Normalizing the features.')
    X = standardize_features(arr_features_to_normalize, feature_vector)[arr_features_to_normalize]

    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=29)
    gmm.fit(X)

    print('Using Gaussian Mixture Model to classify products into 3 categories:\n1.Stable\n2.Trending\n3.Volatile')
    feature_vector['cluster'] = gmm.predict(X)

    probabilities = gmm.predict_proba(X)
    prob_df = pd.DataFrame(probabilities, columns=['stable_prob', 'trending_prob', 'volatile_prob']).round(6)

    final_cat = (pd.concat([feature_vector, prob_df], axis=1))[['product_id', 'cluster']]

    print('Saving the list of categorized products as a csv file...')
    final_cat = (final_cat.merge(
        pd.read_csv(paths['demand_cats']), 
        on='cluster', 
        how='left'
        )
    ).drop(columns='cluster')

    final_cat.to_csv(paths['catzd_products'], index=False)
    print(f'File saved to: {paths['catzd_products']}')
