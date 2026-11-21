import requests
def current_weather(lat,lon):
    u="https://api.open-meteo.com/v1/forecast"
    p={"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"}
    r=requests.get(u,params=p,timeout=10); r.raise_for_status()
    return r.json().get("current",{})
