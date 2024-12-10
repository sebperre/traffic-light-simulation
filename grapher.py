#!/config/workspace/.venv/bin/python3
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Polygon
import json
import os

def list_files_in_folder(folder_path):
    try:
        files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        return files
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
        return []

TIME = "2024-11-27H18M00"

files = list_files_in_folder(f"cache/TOMTOM/{TIME}")

os.mkdir(f"/config/workspace/plots/{TIME}")

for file_name in files:
    data = None

    with open(f"cache/TOMTOM/{TIME}/{file_name}", "r") as file:
        data = json.load(file)

    latitudes = []
    longitudes = []

    for coordinate in data["flowSegmentData"]["coordinates"]["coordinate"]:
        latitudes.append(coordinate["latitude"])
        longitudes.append(coordinate["longitude"])

    # Create a polygon from the points
    polygon_geom = Polygon(zip(longitudes, latitudes))
    polygon_gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon_geom])

    # Project the data to Web Mercator (EPSG:3857) for compatibility with web tiles
    polygon_gdf = polygon_gdf.to_crs(epsg=3857)

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the polygon
    polygon_gdf.plot(ax=ax, alpha=0.5, edgecolor="red", facecolor="none", linewidth=2)

    # Set the extent to the polygon bounds with some padding
    buffer = 400  # in meters
    minx, miny, maxx, maxy = polygon_gdf.total_bounds
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)

    # Add the basemap tiles (streets and buildings)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Remove axis
    ax.set_axis_off()

    # Save the figure as a PNG file
    print(f"Created: {file_name[:-4]}.png")
    plt.savefig(f"plots/{TIME}/{file_name[:-4]}.png", bbox_inches="tight", dpi=300)

