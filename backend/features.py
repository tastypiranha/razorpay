import numpy as np
import pandas as pd
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km

class FeaturePipeline:
    def __init__(self):
        self.category_fraud_rate = {}
        self.category_avg_amt = {}
        self.customer_avg_amt = {}
        self.global_fraud_rate = 0.0
        self.global_avg_amt = 70.0
        self.known_pairs = set()
        self.merchant_stats = {}
        self.fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Compute all target encodings and relational priors strictly from training data (no leakage)."""
        self.global_fraud_rate = float(train_df['is_fraud'].mean())
        self.global_avg_amt = float(train_df['amt'].mean())
        
        # Category fraud rate & average amount
        cat_stats = train_df.groupby('category').agg(
            fraud_rate=('is_fraud', 'mean'),
            avg_amt=('amt', 'mean')
        ).to_dict(orient='index')
        
        self.category_fraud_rate = {cat: data['fraud_rate'] for cat, data in cat_stats.items()}
        self.category_avg_amt = {cat: data['avg_amt'] for cat, data in cat_stats.items()}

        # Customer average spending
        if 'cc_num' in train_df.columns:
            cust_stats = train_df.groupby('cc_num')['amt'].mean().to_dict()
            self.customer_avg_amt = cust_stats

        # Merchant stats (frequency & historical fraud rate)
        merch_stats = train_df.groupby('merchant').agg(
            count=('amt', 'count'),
            fraud_rate=('is_fraud', 'mean'),
            avg_amt=('amt', 'mean')
        ).to_dict(orient='index')
        self.merchant_stats = merch_stats

        # Known customer-merchant pairs
        if 'cc_num' in train_df.columns and 'merchant' in train_df.columns:
            pairs = zip(train_df['cc_num'], train_df['merchant'])
            self.known_pairs = set(pairs)
            
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all signals for tabular/relational model."""
        data = pd.DataFrame(index=df.index)
        
        # Transaction Amount & Log
        amt = df['amt'].astype(float).values
        data['amt'] = amt
        data['amt_log'] = np.log1p(amt)
        
        # Dates and Times
        if 'trans_date_trans_time' in df.columns:
            t_time = pd.to_datetime(df['trans_date_trans_time'])
        elif 'unix_time' in df.columns:
            t_time = pd.to_datetime(df['unix_time'], unit='s')
        else:
            t_time = pd.Timestamp.now()
            
        hours = t_time.dt.hour.values
        data['hour_of_day'] = hours
        data['day_of_week'] = t_time.dt.dayofweek.values
        data['is_night'] = ((hours >= 22) | (hours <= 4)).astype(int)
        
        # Cyclic hour features
        data['sin_hour'] = np.sin(2 * np.pi * hours / 24.0)
        data['cos_hour'] = np.cos(2 * np.pi * hours / 24.0)
        
        # Age
        if 'dob' in df.columns:
            dob = pd.to_datetime(df['dob'])
            data['age'] = (t_time - dob).dt.days // 365
        else:
            data['age'] = 45 # default average
            
        # Geographic Distance
        if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
            dist = haversine_distance(
                df['lat'].values, df['long'].values,
                df['merch_lat'].values, df['merch_long'].values
            )
            data['geo_distance_km'] = dist
            data['geo_dist_log'] = np.log1p(dist)
        else:
            data['geo_distance_km'] = 0.0
            data['geo_dist_log'] = 0.0

        # Gender binary
        if 'gender' in df.columns:
            data['gender_M'] = (df['gender'] == 'M').astype(int)
        else:
            data['gender_M'] = 0
            
        # City Population Log
        if 'city_pop' in df.columns:
            data['city_pop_log'] = np.log1p(df['city_pop'].fillna(1000).clip(lower=0).values)
        else:
            data['city_pop_log'] = 8.0

        # Category encodings (Target Encoded & Ratio)
        if 'category' in df.columns:
            data['category_fraud_rate'] = df['category'].map(self.category_fraud_rate).fillna(self.global_fraud_rate).values
            cat_avg = df['category'].map(self.category_avg_amt).fillna(self.global_avg_amt).values
            data['amt_to_cat_avg'] = amt / (cat_avg + 1e-5)
        else:
            data['category_fraud_rate'] = self.global_fraud_rate
            data['amt_to_cat_avg'] = 1.0

        # Customer Spending Ratio
        if 'cc_num' in df.columns:
            cust_avg = df['cc_num'].map(self.customer_avg_amt).fillna(self.global_avg_amt).values
            data['amt_to_cust_avg'] = amt / (cust_avg + 1e-5)
        else:
            data['amt_to_cust_avg'] = 1.0

        # Relational / Graph features
        if 'merchant' in df.columns and 'cc_num' in df.columns:
            pairs = list(zip(df['cc_num'], df['merchant']))
            data['is_new_pair'] = [0 if p in self.known_pairs else 1 for p in pairs]
            
            data['merch_freq_log'] = df['merchant'].apply(
                lambda m: np.log1p(self.merchant_stats.get(m, {}).get('count', 0))
            ).values
            data['merch_fraud_rate'] = df['merchant'].apply(
                lambda m: self.merchant_stats.get(m, {}).get('fraud_rate', self.global_fraud_rate)
            ).values
        else:
            data['is_new_pair'] = 0
            data['merch_freq_log'] = 0.0
            data['merch_fraud_rate'] = self.global_fraud_rate

        return data
