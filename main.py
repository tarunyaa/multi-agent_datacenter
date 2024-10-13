from simulation import Simulation, EIADataFetcher

if __name__ == "__main__":
    num_gpus = 4
    num_episodes = 1000
    eia_api_key = "YOUR_EIA_API_KEY"
    
    # Fetch EIA data
    eia_data = EIADataFetcher(api_key=eia_api_key)
    electricity_data = eia_data.fetch_electricity_data()
    # Can use predictive ML to forecase electricty prices as well - import a model

    # Run simulation
    simulation = Simulation(num_gpus, num_episodes)
    simulation.run()
