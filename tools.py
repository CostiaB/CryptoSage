import difflib
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
from langchain_core.tools import tool
import numpy as np

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
        return None, None, f"Can't get the historical data for {coin_name}. Error: {str(err)}"


@tool
def calculate_rsi(coin_name, window_length=14):
    """
    Args:
        prices (no.array): Prices of a coin
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



'''
coins = get_historical_data("bitcoin")
print(coins)


prices (np.array): all prices in a year
'''