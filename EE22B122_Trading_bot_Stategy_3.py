from Strategy import StrategyBase
import numpy as np
import xgboost as xgb

class UserStrategy(StrategyBase):
    
    def __init__(self):
        # Initialize the XGBoost model
        self.model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
        self.is_trained = False
        self.previous_winners = []
        self.previous_second_highest_bids = []
        self.safety_margin = 0.02  # Ensure profit margin
        self.aggressiveness = 0.1  # Adjust aggressiveness for bidding
        
    def update_data(self, winner, second_highest_bid):
        """Update historical data with new round results."""
        self.previous_winners.append(winner)
        self.previous_second_highest_bids.append(second_highest_bid)
        
        # Keep only the last 100 rounds of data
        if len(self.previous_winners) > 100:
            self.previous_winners.pop(0)
            self.previous_second_highest_bids.pop(0)
    
    def train_model(self):
        """Train the XGBoost model if there is enough data."""
        if len(self.previous_winners) >= 20:  # Require at least 20 rounds of data
            X_train = np.array(self.previous_winners).reshape(-1, 1)
            y_train = np.array(self.previous_second_highest_bids)
            
            # Train the model
            self.model.fit(X_train, y_train)
            self.is_trained = True
    
    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        This function makes a bid for the auction:
        1. Predicts the second-highest bid using XGBoost.
        2. Ensures that the bid allows for profit while keeping capital in mind.
        3. The strategy minimizes risk for both winner and second-highest bidder.
        '''
        # Update data with latest round results
        if previous_winners and previous_second_highest_bids:
            self.update_data(previous_winners[-1], previous_second_highest_bids[-1])
        
        # Train the model
        self.train_model()
        
        # Step 1: Predict second-highest bid using XGBoost
        if self.is_trained:
            predicted_second_highest = self.model.predict(np.array([[current_value]]))[0]
        else:
            # Fallback if not enough data to train the model
            predicted_second_highest = current_value * 0.75
        
        # Step 2: Adjust bid based on predicted second-highest bid and ensure profit
        if capital > current_value * 0.5:
            # More aggressive if capital is high: bid above second-highest prediction
            target_bid = predicted_second_highest * (1 + self.aggressiveness)
        else:
            # Conservative bidding if capital is low
            target_bid = predicted_second_highest * (1 - self.safety_margin)

        # Step 3: Ensure positive profit for the winner
        bid = min(target_bid, capital)  # Bid must not exceed available capital
        bid = max(0, bid)  # Ensure bid is not negative
        bid = min(bid, current_value)  # Ensure bid is not more than player's value
        
        # Step 4: Adjust to avoid negative profit for second-highest bidder
        if current_value - bid <= 0:
            bid = current_value * (1 - self.safety_margin)

        return bid