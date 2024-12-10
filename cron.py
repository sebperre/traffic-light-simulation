#!/config/workspace/.venv/bin/python3
import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz

def fetch_osm():
    URL_OVERPASS = "https://overpass-api.de/api/interpreter"
    # Has to be of the form Latitude 1,Longitude 1, Latitude 2, Longitude 2
    AREA_TO_SEARCH = "48.403381,-123.395804,48.447075,-123.313201" # Victoria
    TRAFFIC_STOP_AMOUNTS = 10
    file_cache_format = AREA_TO_SEARCH.replace(',', '.')
    data = None

    if os.path.isfile(f"/config/workspace/cache/OSM/{file_cache_format}.txt"):
        print("getting cached")
        with open(f"/config/workspace/cache/OSM/{file_cache_format}.txt", "r") as file:
            data=json.loads(file.read())
    else:
        print("getting from overpass api")
        # queryCoords=f'''
        # [out:json][timeout:25];
        # (
        # node["highway"="traffic_signals"]({AREA_TO_SEARCH});
        # )->.signals;
        # way(bn.signals);
        # node(w);
        # out body;
        # '''
        query = f"""
        [out:json][timeout:25];
        node["highway"="traffic_signals"]({AREA_TO_SEARCH});
        out body {TRAFFIC_STOP_AMOUNTS};
        >;
        out skel qt;
        """

        data = requests.get(URL_OVERPASS, params={'data': query})
        if data.status_code == 200:
            print("OSM API is reachable!")
            data = data.json()
        else:
            print(data.text)
            print(f"Error in OSM: {data.status_code}")
            exit()

        with open(f"/config/workspace/cache/OSM/{file_cache_format}.txt", "w") as file:
            json.dump(data, file,indent=4)

    return data

def parse_osm(dataCoords,dataLanes):
    AREA_TO_SEARCH = "48.403381,-123.395804,48.447075,-123.313201" # Victoria
    TRAFFIC_STOP_AMOUNTS = 10
    file_cache_format = AREA_TO_SEARCH.replace(',', '.')

    # Convert lanes data into a dictionary keyed by (type, id)
    lanes_dict = {}
    for elem in dataLanes.get('elements', []):
        key = (elem['type'], elem['id'])
        lanes_dict[key] = elem

    # Prepare the final data structure
    # We'll combine elements by matching (type, id)
    combined_elements = []

    # Note: dataCoords contains nodes and ways. 
    # The ways are what correspond to the lanes. The nodes are connected but may not have 'lanes' themselves.
    # Typically, you want to join ways that have lane info to their coordinates. Those ways appear in both sets.
    # Filter ways from dataCoords and join them.
    coords_ways = [e for e in dataCoords.get('elements', []) if e['type'] == 'way']
    for elem in coords_ways:
        key = (elem['type'], elem['id'])
        if key in lanes_dict:
            # Merge tags from lanes_dict and coords from this element
            merged = {**lanes_dict[key], **elem}
            combined_elements.append(merged)

    # If you also need to incorporate nodes, you might store them separately or match them in a similar manner.

    # Construct final data object
    data = {k: v for k, v in dataLanes.items() if k != 'elements'}
    data['elements'] = combined_elements

    # Save to file
    with open(f"/config/workspace/cache/OSM/{file_cache_format}.txt", "w") as file:
        json.dump(data, file, indent=4)

    print("Data successfully merged and saved!")

def tomtom():
    print("getting from tomtom api")
    # Reads data from tomtom and caches it
    # https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
    load_dotenv()
    SECRET_KEY = os.getenv("TOMTOM_API_KEY")

    pst_timezone = pytz.timezone('America/Los_Angeles')
    pst_time = datetime.now(pst_timezone)
    formatted_time = pst_time.strftime("%Y-%m-%dH%HM%M")

    traffic_lights = data["elements"]

    traffic_lights.append({"lat": 40.748454, "lon": -73.984565}) # New York
    traffic_lights.append({"lat":48.854664, "lon": 2.295528}) # Paris
    traffic_lights.append({"lat":-22.905770, "lon":-43.177457}) # Rio De Jenario

    os.mkdir(f"/config/workspace/cache/TOMTOM/{formatted_time}")

    for traffic_light in traffic_lights:
        file_tomtom_cache = f"{traffic_light["lat"]}.{traffic_light["lon"]}.{formatted_time}"
        if os.path.exists(f"/config/workspace/cache/TOMTOM/{file_tomtom_cache}.txt"):
            with open(f"/config/workspace/cache/TOMTOM/{file_tomtom_cache}.txt", "r") as file:
                data = json.load(file)
        else:
            url_tomtom = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/22/json?key={SECRET_KEY}&point={traffic_light["lat"]},{traffic_light["lon"]}&thickness=1"
            response = requests.get(url_tomtom)
            if response.status_code == 200:
                print(f"TOMTOM API Request at {formatted_time} for ({traffic_light["lat"]}, {traffic_light["lon"]})")
                data = response.json()
                with open(f"/config/workspace/cache/TOMTOM/{formatted_time}/{file_tomtom_cache}.txt", "w") as file:
                    json.dump(data, file, indent=4)
            else:
                print(response.text)
                print(f"Error in TOMTOM: {response.status_code}")

def main():
    fetch_osm() # intersections rarely change
    # parse_osm(nodes,ways)
    
    # tomtom()

if __name__ == '__main__':
    main()