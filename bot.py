
import os

with open('openai', 'r') as file:
    OPENAI_API_KEY = file.readline().strip()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
import os
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Убедитесь, что ваш OpenAI API ключ установлен в переменной окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("API ключ OpenAI не установлен. Установите переменную окружения OPENAI_API_KEY.")

# Создаем экземпляр ChatOpenAI с моделью gpt-3.5-turbo
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)


system_template = "Ты помощник, который распознает название криптовалюты из текста пользователя."
human_template = "Какая криптовалюта упоминается в следующем тексте: {text}?"


system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Интеграция с CoinGecko для получения актуальных данных
cg = CoinGeckoAPI()


# Определяем класс для бота
class CryptoBot:
    def __init__(self, llm, chat_prompt):
        self.llm = llm
        self.chat_prompt = chat_prompt

    def get_info(self, coin_name):
        # Получение общей информации о монете
        response = self.llm([
            SystemMessage(content="Ты помощник, который предоставляет информацию о криптовалютах."),
            HumanMessage(content=f"Дай общую информацию о {coin_name}.")
        ])

        # Получение актуальной цены из CoinGecko
        try:
            price_data = cg.get_price(ids=coin_name.lower(), vs_currencies='usd')
            price = price_data[coin_name.lower()]['usd']
            price_info = f"Текущая цена {coin_name}: ${price}"
        except Exception as e:
            price_info = f"Не удалось получить цену для {coin_name}. Ошибка: {str(e)}"

        return response.content, price_info

    def get_crypto_summary(self, user_input):
        # Создаем сообщение на основе пользовательского ввода
        messages = self.chat_prompt.format_messages(text=user_input)

        # Выполняем запрос к модели для извлечения названия монеты
        response = self.llm(messages)
        extracted_text = response.content.strip()

        # Простое предположение, что модель может явно указать монету
        coin_name_candidates = ["bitcoin", "ethereum", "litecoin", "ripple", "dogecoin"]  # Пример монет
        coin_name = None
        for coin in coin_name_candidates:
            if coin.lower() in extracted_text.lower():
                coin_name = coin
                break

        if coin_name is None:
            return f"Не удалось распознать криптовалюту из текста: {user_input}. Попробуйте использовать другое название."

        general_info, price_info = self.get_info(coin_name)
        return f"Информация о {coin_name.capitalize()}:\n{general_info}\n{price_info}"


# Инициализируем бота с созданной моделью и шаблоном
bot = CryptoBot(llm=llm, chat_prompt=chat_prompt)

# Получаем информацию по криптовалюте
if __name__ == "__main__":
    user_input = "эфир"  # Пример пользовательского ввода
    summary = bot.get_crypto_summary(user_input)
    print(summary)
