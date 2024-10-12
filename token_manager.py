class TokenManager:
    def __init__(self, num_gpus, total_tokens):
        self.num_gpus = num_gpus
        self.total_tokens = total_tokens
        self.tokens = np.ones(num_gpus) * (total_tokens / num_gpus)  # Equal distribution initially

    def distribute_tokens(self, requests):
        """Distribute power tokens based on requests and current power supply."""
        total_request = sum(requests)
        if total_request <= self.total_tokens:
            self.tokens = requests
        else:
            # Scales requests proportionally if power requests exceed available tokens
            scaling_factor = self.total_tokens / total_request
            self.tokens = [req * scaling_factor for req in requests]
        return self.tokens

    def dynamic_pricing(self, scarcity_factor):
        """Adjust the token value based on scarcity of renewable energy."""
        return 1.0 / scarcity_factor  # Higher scarcity, higher price
