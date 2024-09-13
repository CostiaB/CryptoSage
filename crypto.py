import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os

with open('openai', 'r') as file:
    OPENAI_API_KEY = file.readline().strip()
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

import os
from pycoingecko import CoinGeckoAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence

# Убедитесь, что у вас установлен ваш OpenAI API ключ в переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("API ключ OpenAI не установлен. Установите переменную окружения OPENAI_API_KEY.")

# Создаем экземпляр ChatOpenAI с моделью gpt-3.5-turbo
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)

# Шаблон для запроса (можно менять структуру)
prompt_template = ChatPromptTemplate.from_template("Дай общую информацию о {coin}.")

# Создаем цепочку с использованием ChatPromptTemplate и LLM
chain = prompt_template | llm

# Интеграция с CoinGecko для получения актуальных данных
cg = CoinGeckoAPI()


# Определяем класс для бота
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


# Инициализируем бота с созданной цепочкой
bot = CryptoBot(chain=chain)

# Получаем информацию по криптовалюте Bitcoin
if __name__ == "__main__":
    coin_name = "Bitcoin"
    general_info, price_info = bot.get_info(coin_name)
    print(f"Информация о {coin_name}: {general_info}")
    print(price_info)
