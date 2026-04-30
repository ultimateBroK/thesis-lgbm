# Glossary

> Simple definitions for all technical terms used in this project.

---

## A

**ATR (Average True Range)**
A number that shows how much a price typically moves in one time period. If ATR = 5, the price usually moves about $5 per hour. High ATR means the market is volatile (moving a lot). Low ATR means the market is calm.

**ATR Multiplier**
A number you multiply by ATR to set how far your take-profit or stop-loss should be. If ATR = 5 and multiplier = 1.5, then TP/SL distance = $7.50.

**Ablation Study**
An experiment where you remove parts of your model one at a time to see which parts are actually helping. Like testing a car without its turbo to see how much the turbo really adds.

---

## B

**Backtest**
A simulation of how a trading strategy would have performed in the past. You pretend to trade using historical data and see if you would have made money. It is not a guarantee of future results.

**Base Learner**
A model that serves as a building block in a stacking ensemble. In this project, GRU and LightGBM are base learners whose predictions feed into the meta-learner.

**Batch Size**
The number of data samples processed at once during neural network training. A batch size of 64 means the model looks at 64 examples, updates its weights, then looks at the next 64.

**Bid / Ask**
Two prices you always see in trading. The **bid** is the highest price a buyer is willing to pay. The **ask** is the lowest price a seller is willing to accept. The difference between them is the spread.

---

## C

**CFD (Contract for Difference)**
A financial product that lets you bet on price movements without owning the actual asset. You can go long (bet on price going up) or short (bet on price going down). You trade on margin with leverage.

**Calmar Ratio**
Total return divided by maximum drawdown. A Calmar ratio of 2.0 means you made twice as much as your worst drop. Higher is better.

**Class Weight**
A technique to handle imbalanced data. If you have 60% "Flat" labels and 20% "Long" and 20% "Short", class weights tell the model to pay more attention to the minority classes.

**Commission**
A fee you pay to your broker for each trade. In this project, the commission is $10 per standard lot round-trip (open + close).

**Confidence Threshold**
The minimum predicted probability required before the model takes a trade. In this project, the threshold is 0.60 (60%) — the model only trades when it is at least 60% confident. Set to 0 to disable and trade on all signals.

**Confusion Matrix**
A table that shows how many predictions were correct and how many were wrong. The rows are the actual values, the columns are the predicted values. A perfect model would only have numbers on the diagonal.

**Correlation**
A measure of how two things move together. If correlation = 1.0, they always move in the same direction. If -1.0, they always move in opposite directions. If 0.0, they are unrelated.

---

## D

**Data Leakage**
A bug where information from the future accidentally gets into the training data. For example, using tomorrow's price to predict today's direction. The model looks great during training but fails in real trading.

**Drawdown**
How much your account drops from its highest point. If your account was $120,000 and fell to $96,000, the drawdown is 20%. Maximum drawdown is the worst drop ever recorded.

**Dropout**
A technique to prevent neural networks from memorizing data. During training, it randomly turns off some neurons (like temporarily firing some employees to make the remaining ones more capable).

---

## E

**Early Stopping**
A technique that stops training when the model stops improving on the validation data. Prevents the model from training too long and memorizing the training data.

**EMA (Exponential Moving Average)**
A type of average that gives more weight to recent data. It reacts faster to new information than a simple moving average.

**Embargo**
An extra gap added after the purge zone at data split boundaries. This is an additional safety measure against data leakage.

**Equity Curve**
A line chart showing how your account balance changes over time as you make trades. A good equity curve goes up smoothly. A bad one looks like a rollercoaster.

**Expectancy**
The average amount of money you expect to make (or lose) per trade. Positive expectancy = the strategy is profitable on average.

---

## F

**Feature**
A piece of information the model uses to make a prediction. In this project, RSI, ATR, and MACD are examples of features. Think of features as "clues" the model looks at.

**Feature Importance**
A ranking of which features the model found most useful. If RSI has the highest importance, it means RSI was the most helpful feature for making predictions.

**Forward Fill (ffill)**
A method to fill missing data by using the last known value. If the RSI at 3:00 PM is missing, it uses the RSI from 2:00 PM.

---

## G

**Gradient Boosting**
A machine learning method that builds many simple decision trees, one after another. Each new tree tries to fix the mistakes of the previous trees. LightGBM is a gradient boosting library.

**GRU (Gated Recurrent Unit)**
A type of neural network designed for sequential data (like time series). It has "gates" that decide what information to remember and what to forget. It is similar to LSTM but simpler and faster.

---

## H

**Hidden States**
The internal representation that a GRU produces after reading a sequence. In this project, the GRU produces 32 numbers (hidden states) that summarize the temporal pattern of the last 48 hours.

**Horizon**
The maximum time window for a trade. In triple barrier labeling, if the price does not hit TP or SL within the horizon (e.g., 24 bars), the trade is closed and labeled as "Flat".

**Hyperparameter**
A setting you choose before training (like learning rate or number of trees). The model cannot learn these from data — you have to set them manually or use auto-tuning (Optuna).

---

## K

**Killer Feature**
Not a real technical term, but in this project, the "killer feature" is the hybrid architecture — combining GRU temporal features with LightGBM's decision-making power.

---

## L

**Label**
The correct answer for each data point. In this project, labels are: +1 (Long/buy), 0 (Flat/hold), -1 (Short/sell). Labels are generated using the Triple Barrier method.

**Leverage**
Borrowed money from your broker. Leverage of 30:1 means with $10,000 in your account, you can trade as if you had $300,000. This project uses 30:1 leverage. Leverage amplifies both profits and losses.

**LightGBM**
A fast, efficient gradient boosting library developed by Microsoft. It builds decision trees and combines their predictions. It handles large datasets well and is widely used in competitions and industry.

**Long**
A trade where you buy, hoping the price will go up. You make money if the price rises.

---

## M

**MACD (Moving Average Convergence Divergence)**
A momentum indicator that shows the relationship between two moving averages. When the MACD line crosses above the signal line, it suggests upward momentum. When it crosses below, it suggests downward momentum.

**Margin Call**
When your broker warns you that your account does not have enough money to maintain your open positions. If you do not add funds, the broker will close your positions.

**Max Drawdown**
The largest percentage drop from a peak to a trough in your equity curve. It measures the worst-case loss you would have experienced.

**Meta-learner**
The second-level model in a stacking ensemble that learns how to best combine the predictions of base learners. In this project, a LightGBM classifier serves as the meta-learner.

---

## N

**NumPy**
A Python library for numerical computing. It handles arrays and mathematical operations very fast.

---

## O

**OHLCV**
Open, High, Low, Close, Volume — the five pieces of information in each candlestick bar. Open = starting price, High = highest price, Low = lowest price, Close = ending price, Volume = number of trades.

**Optuna**
A Python library for automatic hyperparameter tuning. Instead of guessing the best settings, Optuna tries many combinations and finds the ones that give the best results.

**Overfitting**
When a model learns the training data too well, including the noise. It performs great on training data but poorly on new, unseen data. Like a student who memorized exam answers but cannot solve new problems.

**OOF (Out-of-Fold) Predictions**
Predictions made on data that the model was NOT trained on during a specific walk-forward window. These are the test-slice predictions from each sliding window, concatenated to produce an unbiased evaluation dataset.

---

## P

**Parquet**
A file format for storing tabular data efficiently. It is much faster to read and write than CSV, especially for large datasets.

**Pip**
The smallest price movement in a currency pair. For XAU/USD, one pip is typically $0.01 per ounce.

**Pivot Points**
Support and resistance levels calculated from the previous day's high, low, and close prices. Traders use them to identify potential price turning points.

**Polars**
A fast Python library for data manipulation, similar to Pandas but much faster. This project uses Polars for all data processing.

**Profit Factor**
Total profits divided by total losses. A profit factor of 2.0 means you made $2 for every $1 you lost. Above 1.0 = profitable. Below 1.0 = losing money.

**Purge**
A gap at the boundary between training and testing data. It removes data points that could cause leakage because they share information across the split.

---

## R

**R-squared (R²)**
A metric that shows how well a model explains the data. Values range from 0 to 1. Higher is better.

**Recovery Factor**
Total net profit divided by maximum drawdown. It tells you how quickly the strategy recovers from its worst loss.

**RSI (Relative Strength Index)**
A momentum indicator that ranges from 0 to 100. Above 70 suggests the asset is overbought (price may drop). Below 30 suggests oversold (price may rise).

---

## S

**SHAP (SHapley Additive exPlanations)**
A method to explain which features contributed to each prediction. It tells you why the model made a specific decision.

**Sharpe Ratio**
A measure of risk-adjusted return. It tells you how much return you get per unit of risk. A Sharpe of 1.0 is considered decent. Above 2.0 is excellent.

**Short**
A trade where you sell, hoping the price will go down. You make money if the price falls.

**Sliding Window**
A walk-forward validation technique where a fixed-size training window slides forward through time. Each window produces a separate train/test split, and the process generates out-of-fold predictions across the entire dataset.

**Slippage**
The difference between the price you wanted and the price you actually got. In fast markets, you might not get filled at the exact price you wanted.

**Sortino Ratio**
Similar to Sharpe ratio, but it only considers downside risk (losses). It is often preferred because upside volatility (gains) should not be penalized.

**Spread**
The difference between the bid and ask price. This is a cost you pay every time you trade. A spread of 2 pips means you start each trade $2 in the hole.

**Stacking**
An ensemble method that trains multiple base models independently, then uses their predictions as input features for a meta-learner that makes the final prediction. Unlike simple concatenation (hybrid), stacking learns the optimal combination weights from data.

**Stop-Loss (SL)**
A price level where you automatically exit a losing trade to limit your loss. If you buy at $2,000 and set a stop-loss at $1,990, you will exit if the price drops to $1,990.

**Stop-Out**
When your broker forcibly closes your positions because your margin level is too low. This happens when losses are too large relative to your account balance.

---

## T

**Take-Profit (TP)**
A price level where you automatically exit a winning trade to lock in your profit. If you buy at $2,000 and set take-profit at $2,020, you will exit when the price reaches $2,020.

**Tick**
The smallest price increment for an asset. For XAU/USD, one tick = $0.01 per ounce (config: `tick_size = 0.01`). Spread and slippage are measured in ticks.

**Time Series**
Data collected over time in chronological order. Stock prices, temperature readings, and heart rate monitors all produce time series data. Order matters — you cannot shuffle it randomly.

**Triple Barrier Method**
A labeling method that places three "barriers" around the entry price: a take-profit barrier above, a stop-loss barrier below, and a time barrier (horizon). Whichever barrier the price hits first determines the label.

**Train / Validation / Test Split**
Dividing your data into three parts:
- **Train** — Used to teach the model (2018-2022). Raw data starts from January 2013; walk-forward effective training start depends on `min_train_bars` and window parameters — the first window begins after sufficient historical bars accumulate.
- **Validation** — Used to check the model during training (2023)
- **Test** — Used for the final evaluation (2024-2026), never seen during training

---

## V

**Validation Set**
A portion of data used during training to check if the model is learning well. It helps you decide when to stop training (early stopping) and which hyperparameters work best.

---

## W

**Walk-Forward Validation**
A time-series validation method where the model is trained on a historical window and tested on the subsequent period, then the window slides forward. This mimics real-world deployment and prevents look-ahead bias. The default mode in this project.

**Win Rate**
The percentage of trades that made money. A 55% win rate means 55 out of every 100 trades were profitable.

---

## X

**XAU/USD**
The ticker symbol for gold priced in US dollars. XAU is the chemical symbol for gold. USD is the US dollar. It is one of the most traded instruments in the world.
