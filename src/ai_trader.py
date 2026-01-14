import os
import numpy as np
import pandas as pd
import pickle

class AITrader:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = discount_factor      # Discount factor for future rewards
        self.alpha = learning_rate        # Learning rate
        self.epsilon = exploration_rate   # Initial exploration rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        # The Q-table
        self.q_table = np.zeros((state_size, action_size))

    def _get_state(self, row):
        """
        Discretizes the continuous feature values into a single discrete state.
        This is a simple example; more sophisticated state representations can be used.
        """
        # Example state representation:
        # State is based on whether the price is above/below short/long term moving averages
        ma5_vs_ma20 = 1 if row['ma_5'] > row['ma_20'] else 0
        ma20_vs_ma60 = 1 if row['ma_20'] > row['ma_60'] else 0
        
        # RSI states: 0 for oversold, 1 for normal, 2 for overbought
        if row['rsi_14'] < 30:
            rsi_state = 0
        elif row['rsi_14'] > 70:
            rsi_state = 2
        else:
            rsi_state = 1
        
        # Combine into a single integer state
        # This is a simple hashing function for our discrete features
        state = ma5_vs_ma20 * 6 + ma20_vs_ma60 * 3 + rsi_state
        return state

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)  # Explore: choose a random action
        return np.argmax(self.q_table[state])           # Exploit: choose the best known action

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation.
        Q(s, a) = Q(s, a) + alpha * (R + gamma * max_a' Q(s', a') - Q(s, a))
        """
        q_predict = self.q_table[state, action]
        q_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (q_target - q_predict)

    def decay_epsilon(self):
        """
        Reduces the exploration rate over time.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path='models/ai_trader_q_table.pkl'):
        """Saves the Q-table to a file."""
        if not os.path.exists('models'):
            os.makedirs('models')
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path='models/ai_trader_q_table.pkl'):
        """Loads the Q-table from a file."""
        with open(file_path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {file_path}")
        self.epsilon = self.epsilon_min # After loading, assume we are in exploitation mode

# --- Training Helpers ---
def run_episode(data, trader, train=True):
    total_profit = 0
    current_holding = 0 # 0 for no stock, 1 for holding stock
    buy_price = 0

    for i in range(len(data) - 1):
        row = data.iloc[i]
        state = trader._get_state(row)
        action = trader.choose_action(state)

        reward = 0
        # Action 1: Buy
        if action == 1 and current_holding == 0:
            current_holding = 1
            buy_price = row['close']
            reward = -0.001

        # Action 2: Sell
        elif action == 2 and current_holding == 1:
            sell_price = row['close']
            profit = (sell_price - buy_price) / buy_price
            reward = profit
            total_profit += profit
            current_holding = 0
            buy_price = 0

        # Action 0: Hold
        else:
            pass

        next_row = data.iloc[i + 1]
        next_state = trader._get_state(next_row)
        if train:
            trader.learn(state, action, reward, next_state)

    if current_holding == 1 and buy_price > 0:
        final_price = data.iloc[-1]['close']
        profit = (final_price - buy_price) / buy_price
        total_profit += profit

    return total_profit

# --- Training Script ---
if __name__ == '__main__':
    STOCK_CODE = "005930"
    features_data_path = f"data/{STOCK_CODE}_daily_features.csv"

    # Load data
    try:
        data = pd.read_csv(features_data_path)
        print(f"Loaded {len(data)} rows of feature data.")
    except FileNotFoundError:
        print(f"Error: Feature data file not found at {features_data_path}")
        exit()

    data = data.reset_index(drop=True)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)

    # Initialize environment and agent
    # State size is based on our discretization function (2 * 2 * 3 = 12 possible states)
    STATE_SIZE = 12 
    # Actions: 0=Hold, 1=Buy, 2=Sell
    ACTION_SIZE = 3
    trader = AITrader(STATE_SIZE, ACTION_SIZE)
    
    EPISODES = 10 # An "episode" is one full run through the historical data
    
    for episode in range(1, EPISODES + 1):
        total_profit = run_episode(train_data, trader, train=True)

        # Decay exploration rate after each episode
        trader.decay_epsilon()
        
        print(f"Episode: {episode}/{EPISODES} | Total Profit: {total_profit:.4f} | Epsilon: {trader.epsilon:.4f}")

    # Evaluate on holdout data with greedy policy
    prev_epsilon = trader.epsilon
    trader.epsilon = 0.0
    test_profit = run_episode(test_data, trader, train=False)
    trader.epsilon = prev_epsilon
    print(f"Test Profit (holdout): {test_profit:.4f}")

    # Save the trained model
    trader.save_model()
