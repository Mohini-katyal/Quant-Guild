
from Strategy import StrategyBase
import numpy as np
import random
from Strategy import StrategyBase  # Assuming StrategyBase is defined in your setup

class UserStrategy(StrategyBase):
    def __init__(self):
        # Correctly initialize attributes in the constructor
        self.previous_bids = []  # Track past bids for analysis
        self.previous_winners = []
        self.previous_second_highest_bids = []
    
    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Optimized bidding strategy based on Expected Value, Adaptive Bidding, and Capital Management.
        '''
        # Bayes-Nash Equilibrium Strategy: b(x_i) = (n-1)/n * x_i
        bne_bid = (num_bidders - 1) / num_bidders * current_value

        # Default bid based on adaptive learning and risk management
        if previous_winners and previous_second_highest_bids:
            avg_winner = np.mean(previous_winners[-10:])  # Use last 10 rounds
            avg_second_highest = np.mean(previous_second_highest_bids[-10:])
        else:
            avg_winner = current_value * 0.75
            avg_second_highest = current_value * 0.5

        # Adaptive Bid: Based on current value, historical data, and randomness
        adaptive_bid = min(current_value * 0.8, avg_winner + random.uniform(-0.05, 0.05) * current_value)
        
        # Final calculated bid, balancing between Bayes-Nash and adaptive strategy
        base_bid = max(bne_bid, adaptive_bid)

        # Capital Management: Bid only a portion of your capital to stay safe
        safe_bid = min(base_bid, capital * 0.25)

        # Introduce randomness to avoid predictable patterns
        random_factor = random.uniform(0.97, 1.03)
        final_bid = safe_bid * random_factor

        # Capital constraints: Don't bid more than your capital allows
        final_bid = min(final_bid, capital)

        # Keep track of previous bids, winners, and second-highest bids
        self.previous_bids.append(final_bid)
        self.previous_winners = previous_winners
        self.previous_second_highest_bids = previous_second_highest_bids

        return final_bid
