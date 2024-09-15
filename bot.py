
import os

with open('openai', 'r') as file:
    OPENAI_API_KEY = file.readline().strip()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
import os
from pycoingecko import CoinGeckoAPI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import difflib  # To implement fuzzy matching

# Ensure your OpenAI API key is set in the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key is not set. Set the OPENAI_API_KEY environment variable.")

# Create an instance of ChatOpenAI with the gpt-3.5-turbo model
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)

# Prompt template for the chatbot
system_template = "You are an assistant that identifies the name of a cryptocurrency from the user's text."
human_template = "What cryptocurrency is mentioned in the following text: {text}?"

# Create message templates
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Combine the message templates into a single ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Integrate CoinGecko for fetching real-time data
cg = CoinGeckoAPI()


# Define the CryptoBot class
class CryptoBot:
    def __init__(self, llm, chat_prompt):
        self.llm = llm
        self.chat_prompt = chat_prompt
        self.coin_list = self.get_coin_list()

    def get_coin_list(self):
        """Fetch the list of all coins available from CoinGecko"""
        try:
            coins = cg.get_coins_list()
            # Return a list of dicts with coin IDs, names, and symbols for more accurate matching
            return {coin['id']: (coin['name'].lower(), coin['symbol'].lower()) for coin in coins}
        except Exception as e:
            raise RuntimeError(f"Failed to fetch coin list: {str(e)}")

    def get_info(self, coin_name):
        """Get general information and price of the given cryptocurrency."""
        response = self.llm.invoke([
            SystemMessage(content="You are an assistant that provides information about cryptocurrencies."),
            HumanMessage(content=f"Give me general information about {coin_name}.")
        ])

        # Get current price from CoinGecko
        try:
            price_data = cg.get_price(ids=coin_name.lower(), vs_currencies='usd')
            price = price_data[coin_name.lower()]['usd']
            price_info = f"Current price of {coin_name}: ${price}"
        except Exception as e:
            price_info = f"Could not retrieve price for {coin_name}. Error: {str(e)}"

        return response.content, price_info

    def get_crypto_summary(self, user_input):
        """Extract the coin name from user input and fetch its details."""
        messages = self.chat_prompt.format_messages(text=user_input)

        # Call the model to process the message using the `invoke` method
        response = self.llm.invoke(messages)
        extracted_text = response.content.strip().lower()

        # Debugging: Print extracted coin name
        print(f"Extracted coin name: {extracted_text}")

        # Match the extracted text with available coins
        coin_name = self.match_coin(extracted_text)

        if coin_name is None:
            return f"Could not recognize a cryptocurrency from the text: {user_input}. Please try a different name."

        general_info, price_info = self.get_info(coin_name)
        return f"Information about {coin_name.capitalize()}:\n{general_info}\n{price_info}"

    def match_coin(self, extracted_text):
        """Match the extracted text with available coins from CoinGecko using fuzzy matching."""
        # List of all coin names and symbols for matching
        coin_names = {coin_id: (name, symbol) for coin_id, (name, symbol) in self.coin_list.items()}

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


# Initialize the bot with the created model and prompt
bot = CryptoBot(llm=llm, chat_prompt=chat_prompt)

# Example usage
if __name__ == "__main__":
    user_input = "Tell me about PERP"  # Example user input
    summary = bot.get_crypto_summary(user_input)
    print(summary)
