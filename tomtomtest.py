from dotenv import load_dotenv
import os
import requests

load_dotenv()

secret_key = os.getenv("TOMTOM_API_KEY")
url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/22/json?key={secret_key}&point=48.427209,-123.364779"

response = requests.get(url)

if response.status_code == 200:
    print("TOMTOM API Reachable")
    print(response.json())
else:
    print(f"Error: {response.status_code}")