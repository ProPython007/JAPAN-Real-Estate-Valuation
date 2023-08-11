import pickle


name = 'FORECAST_final\\forecasts_Aichi.pkl'
with open(name, 'rb') as forecast_file:
    loaded_forecast = pickle.load(forecast_file)

print(loaded_forecast)
