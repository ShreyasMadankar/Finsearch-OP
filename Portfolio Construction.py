import pandas as pd
import matplotlib.pyplot as plt
import os
# Load historical recession data
recession_data = pd.read_csv(os.path.join(os.getcwd(), 'recession_data.csv'))

# Load asset performance data
asset_data = pd.read_csv(os.path.join(os.getcwd(), 'asset_data.csv'))

# Data preprocessing
asset_data['Date'] = pd.to_datetime(asset_data['Date'])
asset_data.set_index('Date', inplace=True)

# Define portfolio allocation
portfolio_allocation = {
    'Equities': 0.30,
    'Bonds': 0.40,
    'Commodities': 0.15,
    'Real Estate': 0.10,
    'Cash': 0.05
}

# Calculate weighted returns
asset_data['Weighted Return'] = (
    asset_data['Equities'] * portfolio_allocation['Equities'] +
    asset_data['Bonds'] * portfolio_allocation['Bonds'] +
    asset_data['Commodities'] * portfolio_allocation['Commodities'] +
    asset_data['Real Estate'] * portfolio_allocation['Real Estate'] +
    asset_data['Cash'] * portfolio_allocation['Cash']
)

import numpy as np

# Calculate daily returns
asset_data['Daily Return'] = asset_data['Weighted Return'].pct_change()

# Sharpe Ratio
sharpe_ratio = asset_data['Daily Return'].mean() / asset_data['Daily Return'].std() * np.sqrt(252)

# Sortino Ratio
downside_risk = asset_data[asset_data['Daily Return'] < 0]['Daily Return'].std() * np.sqrt(252)
sortino_ratio = asset_data['Daily Return'].mean() / downside_risk

# Maximum Drawdown
cumulative_returns = (1 + asset_data['Daily Return']).cumprod()
max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()


def monte_carlo_simulation(returns, n_simulations, n_days):
    simulations = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        daily_simulation = np.random.choice(returns, n_days)
        simulations[i, :] = np.cumprod(1 + daily_simulation) - 1
    return simulations

# Run simulation
n_simulations = 1000
n_days = 252
simulated_returns = monte_carlo_simulation(asset_data['Daily Return'].dropna().values, n_simulations, n_days)

# Plot simulation results
plt.figure(figsize=(10, 6))
plt.plot(simulated_returns.T, color='grey', alpha=0.1)
plt.title('Monte Carlo Simulation of Portfolio Returns')
plt.xlabel('Days')
plt.ylabel('Cumulative Returns')
plt.show()

# Load benchmark data
benchmark_data = pd.read_csv('benchmark_data.csv')
benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
benchmark_data.set_index('Date', inplace=True)

# Calculate cumulative returns
portfolio_cumulative = (1 + asset_data['Daily Return']).cumprod()
benchmark_cumulative = (1 + benchmark_data['Return']).cumprod()

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(portfolio_cumulative, label='Portfolio')
plt.plot(benchmark_cumulative, label='Benchmark', linestyle='--')
plt.title('Portfolio vs. Benchmark Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
