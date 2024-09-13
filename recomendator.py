import os
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence


with open('openai', 'r') as file:
    OPENAI_API_KEY = file.readline().strip()


os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key is not installed.")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)

# Prompt template
prompt_template = ChatPromptTemplate.from_template("Give information about {coin}.")

# Chain with ChatPromptTemplate and LLM
chain = prompt_template | llm

cg = CoinGeckoAPI()



class CryptoBot:
    def __init__(self, chain):
        self.chain = chain

    def get_info(self, coin_name):
        # Выполнение запроса через цепочку
        response = self.chain.invoke({"coin": coin_name})

        # Получение актуальной цены из CoinGecko
        try:
            price_data = cg.get_price(ids=coin_name.lower(), vs_currencies='usd')
            price = price_data[coin_name.lower()]['usd']
            price_info = f"Текущая цена {coin_name}: ${price}"
        except Exception as e:
            price_info = f"Не удалось получить цену для {coin_name}. Ошибка: {str(e)}"

        return response, price_info

    def get_historical_data(self, coin_name):
        try:
            # Получаем данные за последние 365 дней
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
            prices = np.array(historical_data['prices'])[:, 1]  # Получаем только цены

            min_price = np.min(prices)
            max_price = np.max(prices)
            change_percentage = ((prices[-1] - prices[0]) / prices[0]) * 100

            return min_price, max_price, change_percentage
        except Exception as e:
            return None, None, f"Не удалось получить исторические данные для {coin_name}. Ошибка: {str(e)}"

    def calculate_rsi(self, prices, window_length=14):
        # Рассчитываем RSI на основе цен
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = -np.where(delta < 0, delta, 0)

        avg_gain = np.mean(gain[:window_length])
        avg_loss = np.mean(loss[:window_length])

        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_trade_signal(self, prices):
        rsi = self.calculate_rsi(prices)
        signal = "Нейтральный"
        if rsi < 30:
            signal = "Покупка (перепроданность)"
        elif rsi > 70:
            signal = "Продажа (перекупленность)"

        return rsi, signal

    def get_crypto_summary(self, coin_name):
        general_info, price_info = self.get_info(coin_name)
        min_price, max_price, change_percentage = self.get_historical_data(coin_name)

        if min_price is not None and max_price is not None:
            historical_info = (f"За последний год минимальная цена: ${min_price:.2f}, "
                               f"максимальная цена: ${max_price:.2f}, "
                               f"изменение цены: {change_percentage:.2f}%")
        else:
            historical_info = change_percentage  # Если произошла ошибка

        # Получаем исторические данные для анализа сигналов
        historical_data = cg.get_coin_market_chart_by_id(coin_name.lower(), vs_currency='usd', days='365')
        prices = np.array(historical_data['prices'])[:, 1]  # Только цены

        # Рассчитываем торговый сигнал
        rsi, signal = self.get_trade_signal(prices)
        trade_signal_info = f"RSI: {rsi:.2f}, Торговый сигнал: {signal}"

        return general_info, price_info, historical_info, trade_signal_info


# Инициализируем бота с созданной цепочкой
bot = CryptoBot(chain=chain)

# Получаем информацию по криптовалюте Bitcoin
if __name__ == "__main__":
    coin_name = "Bitcoin"
    general_info, price_info, historical_info, trade_signal_info = bot.get_crypto_summary(coin_name)
    print(f"Информация о {coin_name}: {general_info}")
    print(price_info)
    print(historical_info)
    print(trade_signal_info)
