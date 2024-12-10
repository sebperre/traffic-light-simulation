import overpy

# Initialize the Overpass API
api = overpy.Overpass()

# Define the area of interest using a bounding box (latitude and longitude)
# Adjust these coordinates to your desired area
south, west, north, east = 51.5, -0.15, 51.52, -0.1  # Example coordinates for a part of London

# Construct the Overpass QL query
query = f"""
[out:json][timeout:25][bbox:{south},{west},{north},{east}];
(
  node["highway"="traffic_signals"];
);
out body;
way(bn);
out body;
"""

# Execute the query
result = api.query(query)

# Create a dictionary to map node IDs to the number of ways they are part of
node_way_counts = {}

# Collect all traffic signal node IDs
traffic_signal_node_ids = set(node.id for node in result.nodes)

# Initialize the counts to zero
for node_id in traffic_signal_node_ids:
    node_way_counts[node_id] = 0

# Iterate over ways and count how many times each node appears
for way in result.ways:
    try:
        for node in way.nodes:
            if node.id in traffic_signal_node_ids:
                node_way_counts[node.id] += 1
    except:
        print("no nodes")
# Output the results
for node_id, count in node_way_counts.items():
    print(f"Traffic light node {node_id} is part of {count} way(s).")
