import random
import requests

eia_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
api_key = "7u9ipw7cMekOHflVsJFVUKVFHlXyhTqnYaO19Aa9"
class TokenManager:
    def __init__(self, base_price):
        self.base_price = base_price
        self.scaling_factor = self.get_renewable_energy_availability() # normalizing token price to 1.0 at the beginning of program
        self.token_price = self.base_price  # Initialize token price

    def update_token_price(self, renewable_energy_availability):
        """Update token pricing based on real renewable energy availability."""
        if renewable_energy_availability == 0:
            renewable_energy_availability = 0.1  # Avoid division by zero
        self.token_price = self.base_price * self.scaling_factor / renewable_energy_availability
        
    def get_token_price(self):
        return self.token_price

    def get_renewable_energy_availability(self):
        """Fetch renewable energy availability using the EIA API."""
        try:
            x_params = '{"frequency":"hourly","data":["value"],"facets":{},"start":null,"end":null,"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":5000}'
            headers = {"X-Params": x_params}
            response = requests.get(eia_url, headers=headers, params={"api_key": api_key})
            response.raise_for_status()
            data = response.json()
            if "response" in data and "data" in data["response"]:
                renewable_data = data["response"]["data"]
                # Aggregate the total renewable energy from the response
                total_renewable_energy = sum(float(item["value"]) for item in renewable_data if "value" in item and item["value"].isdigit())
                return total_renewable_energy
            else:
                print("No renewable energy data available.")
                return 0.1
        except Exception as e:
            print(f"Error fetching renewable energy data: {e}")
            return 0.1