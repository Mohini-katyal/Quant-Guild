from Strategy import StrategyBase
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import random

class CombinedStrategy(StrategyBase):
    def __init__(self):
        # Polynomial Regression model setup
        self.poly_features = PolynomialFeatures(degree=3)  # Increased degree for more accuracy
        self.model = make_pipeline(self.poly_features, LinearRegression())
        self.epsilon = 0.1  # Exploration parameter for MAB
        self.trained = False
    
    def train_model(self, previous_winners, previous_second_highest_bids):
        # Ensure there are enough data points for training
        if len(previous_winners) < 10:
            return
        # Train Polynomial Regression model
        X = np.array(previous_second_highest_bids[-10:]).reshape(-1, 1)
        y = np.array(previous_winners[-10:])
        self.model.fit(X, y)
        self.trained = True

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        if len(previous_winners) < 3:
            return min(capital, 10)  # Default bid if not enough history

        # Step 1: Train the model if enough data is available
        if not self.trained:
            self.train_model(previous_winners, previous_second_highest_bids)

        # Step 2: Polynomial Regression Prediction
        if self.trained:
            predicted_winner = self.model.predict([[previous_second_highest_bids[-1]]])[0]
        else:
            predicted_winner = np.mean(previous_winners)  # Fallback if not enough data

        # Step 3: Moving Average for Trend Stability
        moving_avg = np.mean(previous_winners[-5:])

        # Step 4: Combine both predictions (Polynomial Regression and Moving Average)
        combined_bid = (predicted_winner + moving_avg) / 2

        # Step 5: Multi-Armed Bandit Exploration (explore vs exploit)
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Random bid within a reasonable range based on capital
            bid = random.uniform(0, min(capital, current_value))
        else:
            # Exploitation: Use the combined prediction adjusted by capital
            safe_bid = min(combined_bid - 5, capital * 0.4)  # Capital-aware adjustment
            bid = max(safe_bid, 1)  # Ensure we don't bid too low

        # Step 6: Final bid with capital constraint
        bid = min(bid, capital)

        return bid
