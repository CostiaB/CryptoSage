from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os

with open('openai', 'r') as file:
    OPENAI_API_KEY = file.readline().strip()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


# Шаблон для обработки запроса
prompt = PromptTemplate(
    input_variables=["coin"],
    template="What is the current state of the following cryptocurrency: {coin}?"
)


# Создаем языковую модель
llm = OpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo")

# Создаем цепочку, которая связывает модель с шаблоном
chain = LLMChain(llm=llm, prompt=prompt)

def ask_bot(coin_name):
    # Передаем запрос пользователя в цепочку
    response = chain.run(coin=coin_name)
    return response



from pycoingecko import CoinGeckoAPI

# Инициализация клиента CoinGecko
cg = CoinGeckoAPI()

# Функция получения текущего курса криптовалюты
def get_current_price(coin_id):
    data = cg.get_price(ids=coin_id, vs_currencies='usd')
    return data[coin_id]['usd']

# Функция получения изменения цены за год
def get_price_change(coin_id):
    coin_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=365)
    prices = coin_data['prices']
    start_price = prices[0][1]
    end_price = prices[-1][1]
    price_change = ((end_price - start_price) / start_price) * 100
    return price_change



class CryptoBot:
    def __init__(self, model, coin_api):
        self.chain = LLMChain(llm=model, prompt=prompt)
        self.coin_api = coin_api

    def get_info(self, coin_name):
        # Получаем информацию через LangChain
        response = self.chain.run(coin=coin_name)
        return response

    def get_current_price(self, coin_name):
        return self.coin_api.get_price(ids=coin_name, vs_currencies='usd')[coin_name]['usd']

    def get_price_change_year(self, coin_name):
        coin_data = self.coin_api.get_coin_market_chart_by_id(id=coin_name, vs_currency='usd', days=365)
        prices = coin_data['prices']
        start_price = prices[0][1]
        end_price = prices[-1][1]
        price_change = ((end_price - start_price) / start_price) * 100
        return price_change



# Инициализация модели и API
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
coin_api = CoinGeckoAPI()

# Создаем экземпляр бота
bot = CryptoBot(model=llm, coin_api=coin_api)

# Запрос информации о Bitcoin
general_info = bot.get_info("Bitcoin")
current_price = bot.get_current_price("bitcoin")
price_change_year = bot.get_price_change_year("bitcoin")

print("General Info:", general_info)
print("Current Price of Bitcoin:", current_price)
print("Price Change in Last Year:", price_change_year, "%")
