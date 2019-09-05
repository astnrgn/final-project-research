from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt


# Adding an API Key and connecting to DarkSky.net
# https://api.darksky.net/forecast/[key]/[latitude],[longitude]

# API_KEY = 'ffd516440a43c75cbd9c4e2a477aedad'
BASE_URL = "https://pudding.cool/2017/12/mars-data/marsWeather.json"


# https://pudding.cool/2017/12/mars-data/marsWeather.json
