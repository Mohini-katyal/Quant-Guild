import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Helper functions to create alpha signals
def rank_transform(x):
    """Convert values into their rank (percentile-based)."""
    return pd.Series(x).rank(pct=True).values

def signed_power_transform(x, power):
    """Apply signed power transformation for asymmetrical scaling."""
    return np.sign(x) * (np.abs(x) ** power)

def rolling_correlation(x, y, window_size):
    """Compute rolling correlation between two series with a given window size."""
    return pd.Series(x).rolling(window=window_size).corr(pd.Series(y))

def rolling_stddev(x, window_size):
    """Compute rolling standard deviation for a series over a specified window."""
    return pd.Series(x).rolling(window=window_size).std()

# Load training and testing datasets
# Ensure the correct path to your dataset
downloads_dir = "C:\\Users\\mohin\\Downloads"

# Read data
train_data = pd.read_csv(f"{downloads_dir}\\taramani-quant-research-contest-tqrc\\train_data.csv")
final_test_data = pd.read_csv(f"{downloads_dir}\\taramani-quant-research-contest-tqrc\\final_test_data.csv")

# Handle missing values by replacing NaNs with the column's mean
train_data = train_data.fillna(train_data.mean())
final_test_data = final_test_data.fillna(final_test_data.mean())

# Feature transformations - log-transformed values to linearize the feature space
# Apply log1p transformation (log(1+x)) for better handling of small values
for feature in ["midprice", "last_trade_price", "recent_buy_order_count", "net_open_interest_change", "recent_sell_order_count", "total_order_count"]:
    train_data[f"{feature}_log"] = train_data[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)
    final_test_data[f"{feature}_log"] = final_test_data[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)

# Features with a Gaussian distribution need to be standardized for better model performance
gaussian_features = ["bid_price_1", "bid_price_2", "bid_price_3", "bid_price_4", "bid_price_5", 
                     "ask_price_1", "ask_price_2", "ask_price_3", "ask_price_4", "ask_price_5", 
                     "bid_volume_1", "ask_volume_1", "ask_volume_2", "ask_volume_3", "ask_volume_4",
                     "ask_volume_5", "bid_volume_2", "bid_volume_3", "bid_volume_4", "bid_volume_5"]

# Standardize Gaussian-distributed features using StandardScaler
scaler = StandardScaler()
train_data[gaussian_features] = scaler.fit_transform(train_data[gaussian_features])
final_test_data[gaussian_features] = scaler.transform(final_test_data[gaussian_features])

# Feature Engineering: Custom alpha generation based on financial insights
# Introduce additional alpha features using ratios and differences of existing data
def generate_alphas(df):
    # Alpha 1: Liquidity measure via ratio of bid and ask prices
    df['alpha_1'] = rank_transform(df['bid_price_1'] / (df['ask_price_1'] + 1e-10))
    
    # Alpha 2: Momentum-based alpha using volume-weighted prices
    df['alpha_2'] = rank_transform(df['bid_price_1'] * df['bid_volume_1'] - df['ask_price_1'] * df['ask_volume_1'])
    
    # Alpha 3: Mean reversion calculated via the difference in prices and volumes
    df['alpha_3'] = rank_transform(df['last_trade_price_log'] - df['midprice_log']) * rank_transform(df['recent_buy_order_count_log'] - df['recent_sell_order_count_log'])
    
    # Alpha 4: Buy-sell imbalance based on order count
    df['alpha_4'] = rank_transform(df['recent_buy_order_count_log'] - df['recent_sell_order_count_log'])
    
    # Alpha 5: Open interest change adjusted by total order count
    df['alpha_5'] = rank_transform(df['net_open_interest_change_log'] / (df['total_order_count_log'] + 1e-10))
    
    # Alpha 6: Market activity proxy by summing bid and ask volume changes
    df['alpha_6'] = rank_transform(df[['bid_volume_1', 'ask_volume_1']].sum(axis=1))
    
    return df

# Apply custom alpha generation to both training and testing datasets
train_data = generate_alphas(train_data)
final_test_data = generate_alphas(final_test_data)

# Define the list of features, including the transformed and alpha features
alpha_columns = [f'alpha_{i}' for i in range(1, 7)]
features = ["midprice_log", "last_trade_price_log", "recent_buy_order_count_log", 
            "recent_sell_order_count_log", "net_open_interest_change_log", "total_order_count_log", 
            "bid_price_1", "bid_price_2", "bid_price_3", "bid_price_4", "bid_price_5", 
            "ask_price_1", "ask_price_2", "ask_price_3", "ask_price_4", "ask_price_5", 
            "bid_volume_1", "ask_volume_1", "ask_volume_2", "ask_volume_3", "ask_volume_4", 
            "ask_volume_5", "bid_volume_2", "bid_volume_3", "bid_volume_4", "bid_volume_5"] + alpha_columns

# Target variable
target = "actual_returns"

# Polynomial Regression: Use polynomial terms up to the specified degree
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(train_data[features], train_data[target])

# Make predictions on the test data
final_test_data["predicted_returns"] = model.predict(final_test_data[features])

# Save the predictions in a CSV file for submission
output_file = f"{downloads_dir}/submission_polynomial_alphas.csv"
submission = final_test_data[["timestamp_code", "predicted_returns"]]
submission.to_csv(output_file, index=False)

print(f"Submission file created: {output_file}")

# Visualizing the relationships between features and the target variable
num_plots = len(features) + 1  # Include a plot for the target vs target
rows, cols = (num_plots + 3) // 4, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

# Scatter plots for each feature vs target
for i, feature in enumerate(features):
    axes[i].scatter(train_data[feature], train_data[target], alpha=0.5)
    axes[i].set_title(f"{feature} vs {target}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(target)

# Plot of the target variable against itself
axes[len(features)].scatter(train_data[target], train_data[target], alpha=0.5)
axes[len(features)].set_title(f"{target} vs {target}")
axes[len(features)].set_xlabel(target)
axes[len(features)].set_ylabel(target)

# Hide any unused subplots
for j in range(len(features) + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
