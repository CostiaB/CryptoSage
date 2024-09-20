import difflib
from datetime import datetime, timedelta

import pandas as pd
from pycoingecko import CoinGeckoAPI
from langchain_core.tools import tool
import numpy as np
import pandas_ta as ta
cg = CoinGeckoAPI()


@tool
def get_coin_list():
    """
    Fetch the list of all coins available from CoinGecko

    Returns:
        dict: dict of all coins available from CoinGecko
    """
    try:
        coins = cg.get_coins_list()
        # Return a list of dicts with coin IDs, names, and symbols for more accurate matching
        return {coin['id']: (coin['name'].lower(), coin['symbol'].lower()) for coin in coins}
    except Exception as err:
        raise RuntimeError(f"Failed to fetch coin list: {str(err)}")


@tool
def match_coin(extracted_text, coin_dict):
    """
    Match the extracted text with available coins from CoinGecko using fuzzy matching.
    Args:
        extracted_text str: Message with coin name
        coin_dict dict: Dictionary with all coin names
    Returns:
        closest_name str: Closest name of a crypto ticket
        """
    # List of all coin names and symbols for matching
    coin_names = {coin_id: (name, symbol) for coin_id, (name, symbol) in
                  coin_dict.items()}

    # Try to find a close match using fuzzy matching
    closest_name = None
    highest_ratio = 0.0

    for coin_id, (name, symbol) in coin_names.items():
        # Fuzzy matching for both name and symbol
        name_ratio = difflib.SequenceMatcher(None, extracted_text, name).ratio()
        symbol_ratio = difflib.SequenceMatcher(None, extracted_text, symbol).ratio()

        # Pick the highest match ratio
        if name_ratio > highest_ratio:
            highest_ratio = name_ratio
            closest_name = coin_id
        if symbol_ratio > highest_ratio:
            highest_ratio = symbol_ratio
            closest_name = coin_id

    # Consider a match only if it is above a certain threshold (e.g., 0.7 for similarity)
    if highest_ratio > 0.7:
        return closest_name
    return None


@tool
def get_historical_data(coin_name):
    """
    Gets the data for the last 365 days and calculates percent range of prices
    Args:
        coin_name (str): Crypto coin ticker name
    Returns:
        min_price (float): minimum price in a year
        max_price (float):  maximum price in a year
        change_percentage (float): percent change in a year
        change_percent_from_max (float): percent of change from maximum to current price

    """
    try:
        # Data for the last 366 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        historical_data = cg.get_coin_market_chart_range_by_id(
            id=coin_name.lower(),
            vs_currency='usd',
            from_timestamp=start_timestamp,
            to_timestamp=end_timestamp
        )
        #print(historical_data)
        prices = np.array(historical_data['prices'])[:, 1]  # only prices

        min_price = np.min(prices)
        max_price = np.max(prices)
        change_percentage = ((prices[-1] - prices[0]) / prices[0]) * 100 # year change
        change_percent_from_max = ((prices[-1]- max_price)/max_price) * 100 # change from maximum

        return min_price, max_price, change_percentage, change_percent_from_max
    except Exception as err:
        return f"Can't get the historical data for {coin_name}. Error: {str(err)}"


@tool
def calculate_rsi(coin_name, window_length=14):
    """
    Args:
        window_length (int): Size of window for RSI calculation
        coin_name: (str): Cryptocoin name
    Returns:
         rsi (float): rsi value for a coin
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    historical_data = cg.get_coin_market_chart_range_by_id(
        id=coin_name.lower(),
        vs_currency='usd',
        from_timestamp=start_timestamp,
        to_timestamp=end_timestamp
    )
    prices = np.array(historical_data['prices'])[:, 1]
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)

    avg_gain = np.mean(gain[:window_length])
    avg_loss = np.mean(loss[:window_length])

    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    return rsi


@tool
def get_trade_signal(coin_name):
    """
    Make a trading signal
    Args:
        coin_name (str): Name of a coin
    Returns:
        signal (str): Signal
        rsi (float): RSI value
    """
    rsi = calculate_rsi(coin_name)
    signal = "Neutral"
    if rsi < 30:
        signal = "Buy (oversold)"
    elif rsi > 70:
        signal = "Sell (overbought)"

    return rsi, signal

@tool
def ohlc_values(coin_name):
    """
        Gets the data for the last 180 days and calculates percent range of prices
        Args:
            coin_name (str): Crypto coin ticker name
        Returns:
            prices list(list(float):  open, high, low, close prices for coins


    """
    try:
        # Data for the last 180 days
        historical_data = cg.get_coin_ohlc_by_id(id=coin_name,
                                                vs_currency='usd',
                                                days=180)

        return historical_data
    except Exception as err:
        return  f"Can't get the historical data for {coin_name}. Error: {str(err)}"


@tool
def calculate_fibonacci_levels(coin_name):
    """
    Call this tool only if user directly asks for Fibonacci levels
    Calculate Fibonacci retracement levels for a given list of prices.

    Args:
        coin_name (str): Token name.

    Returns:
        dict: A dictionary containing Fibonacci levels.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        historical_data = cg.get_coin_market_chart_range_by_id(
            id=coin_name.lower(),
            vs_currency='usd',
            from_timestamp=start_timestamp,
            to_timestamp=end_timestamp
        )
        prices = np.array(historical_data['prices'])[:, 1]

        # Find the highest and lowest price in the data
        high = max(prices)
        low = min(prices)
        print(high, low)

        # Calculate Fibonacci retracement levels
        diff = high - low
        levels = {
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50%': high - (diff * 0.500),
            '61.8%': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786),
            'High': high,
            'Low': low
        }
        return levels
    except Exception as err:
        return f"Can't calculate Fibonacci levels for {coin_name}. Error: {str(err)}"

@tool
def find_support_resistance(coin_name, window=100):
    """
    This tool is called for calculation of  actual resistance and support levels.
    Calculate resistance and support level for a given coin.
    window should be less than 100

    Args:
        coin_name (str): Token name.
        window (int): Length of a window to calculate levels

    Returns:
        support (float): Current support level
        resistance (float): Current resistance level

    """
    try:
        # Data for the last 365 days
        data = cg.get_coin_ohlc_by_id(id=coin_name,
                                      vs_currency='usd',
                                      days=365)
    except Exception as err:
        return f"Can't get the historical data for {coin_name}. Error: {str(err)}"

    data = pd.DataFrame(data, columns=["TS", "Open", "High", "Low", "Close"])
    support_levels = []
    resistance_levels = []

    for i in range(window, len(data)):
        window_data = data.iloc[i - window:i]
        support = window_data['Low'].min()
        resistance = window_data['High'].max()
        support_levels.append(support)
        resistance_levels.append(resistance)

    support = support_levels[-1]
    resistance = resistance_levels[-1]

    return support, resistance


@tool
def MACD_Alligator_advice(coin_name):
    """
       This tool is called for generation of trading signals.

       Args:
           coin_name (str): Token name.

       Returns:
           recommendations (str): Trade signals based on MACD and Alligator

       """
    def fetch_data(coin_name):
        # Fetch historical data from Yahoo Finance (use a crypto symbol)
        try:
            # Data for the last 365 days
            data = cg.get_coin_ohlc_by_id(id=coin_name,
                                          vs_currency='usd',
                                          days=365)
            data = pd.DataFrame(data, columns=["TS", "Open", "High", "Low", "Close"])
            return data
        except Exception as err:
            return f"Can't get the historical data for {coin_name}. Error: {str(err)}"

    # Function to calculate MACD
    def calculate_macd(data):
        # Calculate MACD using pandas-ta
        macd = ta.macd(data['Close'])
        data = pd.concat([data, macd], axis=1)
        return data

    # Custom function to calculate the Alligator indicator
    def calculate_alligator(data):
        jaws_period = 13
        teeth_period = 8
        lips_period = 5

        # Calculate moving averages
        data['Jaws'] = data['High'].rolling(window=jaws_period).mean().shift(8)
        data['Teeth'] = data['High'].rolling(window=teeth_period).mean().shift(5)
        data['Lips'] = data['High'].rolling(window=lips_period).mean().shift(3)

        return data

    # Function to generate buy/sell signals based on MACD and Alligator
    def generate_signals(data):
        signals = []

        for i in range(1, len(data)):
            macd_line = data['MACD_12_26_9'].iloc[i]
            macd_signal = data['MACDs_12_26_9'].iloc[i]
            jaws = data['Jaws'].iloc[i]
            teeth = data['Teeth'].iloc[i]
            lips = data['Lips'].iloc[i]

            # MACD Signal
            if macd_line > macd_signal:
                macd_signal_text = "MACD indicates a possible BUY signal."
            else:
                macd_signal_text = "MACD indicates a possible SELL signal."

            # Alligator Signal
            if lips > teeth > jaws:
                alligator_signal_text = "Alligator indicates an UPTREND."
            elif jaws > teeth > lips:
                alligator_signal_text = "Alligator indicates a DOWNTREND."
            else:
                alligator_signal_text = "Alligator is indecisive."

            # Generate combined signal

            recommendation = f"Date: {pd.to_datetime(data.loc[:, 'TS'], unit='ms')[i]}\n{macd_signal_text}\n{alligator_signal_text}"
            signals.append(recommendation)
        return signals

    # Function to simulate chat-based recommendations
    def chat_recommendations(symbol='BTC-USD'):
        # Fetch data
        data = fetch_data(symbol)

        # Calculate indicators
        data = calculate_macd(data)
        data = calculate_alligator(data)

        # Generate recommendations
        signals = generate_signals(data)

        # Simulate chat responses

        #for signal in signals[-1:]:  # Show the last 5 signals for brevity
        #    print("Advice:", signal)
        #    print("-" * 50)
        return signals[-1]

    # Run the chat-based recommendations
    recommendations = chat_recommendations('bitcoin')
    return recommendations
