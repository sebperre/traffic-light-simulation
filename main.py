import os
import json
import re
from typing import List
from decimal import Decimal, ROUND_DOWN
from queue import Queue
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import heapq
from heapq import heappush, heappop
from collections import deque

# Long-Lat for Traffic Lights
OAKFOUL_VIC = "48.426442.-123.3226985"
PANDORAFERN_VIC = "48.426829.-123.34534"
FORTFOUL_VIC = "48.432167.-123.322405"
JOHNSONFERN_VIC = "48.4254427.-123.345458"
OAKMONTEREY_VIC = "48.4264785.-123.3140903"
OAKEGLIN_VIC = "48.4265087.-123.3193121"
PANDORAFORT_VIC = "48.4265366.-123.3368677"
PANDORAJOHNSON_VIC = "48.4267701.-123.3426018"
GOLDSMITHFOUL_VIC = "48.4304871.-123.3225101"
CADBOROFLOR_VIC = "48.4323992.-123.3209411"
AVENIDARIOBRANCO_RIO = "-22.90577.-43.177457"
WEST34TH_NY = "40.748454.-73.984565"
AVESUFFREN_PARIS = "48.854664.2.295528"

# Constants
DIST_BETWEEN_CARS = 8.96 # meters
AVG_CAR_LENGTH = 4.48 # meters

class Lane:
    def __init__(self):
        pass
class Approach:
    def __init__(self, lanes:List[Lane]):
        self.lanes=lanes
class Intersection:
    def __init__(self, approaches:List[Approach]):
        self.approaches=approaches

def read_tomtom_data(date_times: list = [], intersections = [], error_on_not_found: bool = True) -> dict:
    '''
    Reads TomTom data based on a list of dates and hours. The data is read into a dictionary
    object, and traffic light coordinates are added.

    Parameters:
    date_times (list): A list of date-time values (dates and hours) for which the data should 
                        be fetched. If date_times is empty, it is preceived as wanting all dates and times.
    intersections (list): A list of strings of long and lat sepearted by a . for which the data should 
                        be fetched. If intersections is empty, it is preceived as wanting all intersections available.
    error_on_not_found (bool): If True, an error is raised when the data is not found. If False, 
                               the function proceeds without raising an error.

    Returns:
    dict: A dictionary containing the TomTom data and traffic light coordinates.
    '''
    # Checks to see if TomTom data exists
    cwd = os.getcwd()
    if not os.path.isdir('./cache/TOMTOM'):
        print('couldnt access ./cache/TOMTOM!')
        exit(1)
    os.chdir('./cache/TOMTOM')
    data = {}

    # If we provide no date_times then it will search all directories
    if not date_times:
        for dir in os.listdir('.'):
            if os.path.isdir(dir):
                date_times.append(dir)

    # Loop through directories and fetch all the intersections requested
    for date_time in date_times:
        if os.path.isdir(f'./{date_time}'):
            os.chdir(f'./{date_time}')
            if date_time not in data.keys():
                    data[date_time] = {}
            # If we provide no intersections then it will fetch all intersections
            if not intersections:
                intersections = os.listdir('.')
            else:
                # Rename all intersections with the date at the end because
                # that's how the file is structured
                for i, intersection in enumerate(intersections):
                    intersections[i] = intersection + '.' + date_time + '.txt'
            # Loop through all the files provided and reads the contents into the dict and adds the coords
            for intersection in intersections:
                if os.path.isfile(intersection):
                    with open(intersection, 'r') as f:
                        contents = f.read()
                        if len(contents)==0:
                            continue
                        try:
                            entry = json.loads(contents) # also coords
                            r = re.findall(r'(-?\d+\.\d+)', intersection)
                            entry['flowSegmentData']['traffic_light_coord'] = {'lat': r[0], 'long': r[1]}
                        except:
                            print('parse fail', contents)
                            exit(1)
                        if intersection not in data[date_time].keys():
                            data[date_time][intersection] = entry['flowSegmentData']
                        else:
                            print(f"WARN: got duplicate data->'${date_time}'->'${intersection}'")
                else:
                    if error_on_not_found:
                        print(f'Error: Couldnt find {intersection}')
                        exit(1)
                    else:
                        print(f"Warning: Couldn't find {intersection}")
            os.chdir('..')
        else:
            if error_on_not_found:
                print(f'Error: Couldnt find {date_time}')
                exit(1)
            else:
                print(f"Warning: Couldn't find {date_time}")
    os.chdir(cwd)
    return data

def flatten(data:{})->[{}]:
    '''
    flattens data->date->entry into data->entry,
    containing the entry plus date and coords.
    returns a list of dicts

    eg. (will fail due to formatting)
    >> flatten(read_tomtom_data())[0]
    {
        'frc': 'FRC5', 
        'currentSpeed': 45, 
        'freeFlowSpeed': 45, 
        'currentTravelTime': 125, 
        'freeFlowTravelTime': 125, 
        'confidence': 1, 
        'roadClosure': False, 
        'coordinates': {
            'coordinate': [
                ...
            ]
        }, 
        '@version': 'traffic-service-flow 1.0.120', 
        'date': '2024-12-01H21M00', 
        'coords': ('48.432167', '48.432167')
    }
    '''
    flat=[]
    for date in data.keys():
        for entry in data[date].keys():
            r = re.findall(r'(-?\d+\.\d+)', entry)
            coords=(float(r[0]),float(r[1]))
            data[date][entry]['date']=date
            data[date][entry]['coords']=coords
            flat.append(data[date][entry])
    return flat

def read_intersections():
    with open('intersection_lanes.json','r') as f:
        data=json.loads(f.read())
        f.close()
    return data

def simulate_arrivals(queues, lambdas, num_events): # lanes
    '''
    for each lane per approach
    '''
    assert(len(queues)==len(lambdas))
    for i,approach in enumerate(queues):
        assert(len(approach)==len(lambdas[i]))
        # for j,lane in enumerate(approach):
        for j in range(len(approach)):
            # print(j)
            lam = lambdas[i][j]
            inter_arrival_times = np.random.exponential(1 / lam, num_events)
            arrival_times = np.cumsum(inter_arrival_times)
            approach[j]=arrival_times # overwrite lane as it should be empty at this point

def merge_queues_with_priority(approaches): 
    '''
    merge approaches->lanes->... into one big PQ
    '''
    merged_queue = []
    for i,approach in enumerate(approaches):
        for j,lane in enumerate(approach):
            for arrival_time in lane:
                heapq.heappush(merged_queue, (arrival_time,i))  # (time, queue index)
    return merged_queue


# def simulate_intersection(merged_arrival_times):
#     AT = 0 # constants
#     LANE = 1
#     ID = 2

#     VEHICLE_PROCESSING_TIME = 2

#     current_time = 0
#     arrival_events = [] # priorityqueue, contains (AT,queue)
#     processing_queue = []

#     # times since epoch
#     queue_arrival_times={}
#     queue_leave_times={} # arrive at intersection
#     processed_times={} # leave intersection

#     num_vehicles = 0 # serves as ID
#     for arrival_time, lane in merged_arrival_times:
#         heappush(arrival_events, (arrival_time, lane, num_vehicles))
#         num_vehicles = num_vehicles+1

#     while arrival_events or processing_queue:
#         if not processing_queue: # skip time forward to event 0
#             current_time = arrival_events[0][AT]
        
#         # Process arrival events that are due
#         # while
#         if arrival_events and arrival_events[0][AT]<=current_time:
#             event_time,lane,id = heappop(arrival_events)
#             heappush(processing_queue, (current_time,lane,id))
            
#             # set arrival times
#             queue_arrival_times[id] = event_time

#             # set leave queue time
#             queue_leave_times[id] = current_time

#         # Process the vehicle in the processing queue
#         if processing_queue:
#             start_time,lane,id = processing_queue.pop(0)
#             processing_time = max(current_time,start_time) + VEHICLE_PROCESSING_TIME
            
#             assert(current_time==queue_leave_times[id])

#             # set process time (leave intersection)
#             processed_times[id] = processing_time

#             current_time = processing_time  # Update current time to reflect the vehicle's departure time
    
#     return queue_arrival_times,queue_leave_times,processed_times

from heapq import heappop, heappush
from collections import deque

def simulate_intersection_fixed(merged_arrival_times):
    '''
    fixed interval light intersection
    logic: alternate between north-south, east-west queues.
    '''
    AT = 0  # arrival time index
    LANE = 1
    ID = 2

    VEHICLE_PROCESSING_TIME = 2  # seconds per vehicle
    LIGHT_CYCLE_DURATION = 10  # seconds

    def get_lane_group(lane):
        if lane in (0,1,4,5): # these are north-south lanes
            return "north_south"
        elif lane in (2,3,6,7): # these are east-west lanes
            return "east_west"

    current_time = 0
    arrival_events = [] # PQ
    queues = { 
        "north_south": deque(),
        "east_west": deque()
    }

    # metrics
    queue_arrival_times = {}
    queue_leave_times = {}
    processed_times = {}

    num_vehicles = 0 # ID
    for arrival_time, lane in merged_arrival_times:
        heappush(arrival_events, (arrival_time,lane,num_vehicles))
        num_vehicles += 1

    # Traffic light management
    light_group = "north_south" # start with this
    next_light_switch = LIGHT_CYCLE_DURATION

    while arrival_events or queues["north_south"] or queues["east_west"]:
        # Advance time to the next event if necessary
        if not (queues["north_south"] or queues["east_west"]):
            if arrival_events:
                current_time = max(current_time, arrival_events[0][AT])
            else:
                break  # No vehicles left to process

        # Process arrival events
        while arrival_events and arrival_events[0][AT] <= current_time:
            event_time, lane, id = heappop(arrival_events)
            group = get_lane_group(lane)
            queues[group].append((lane, id))
            queue_arrival_times[id] = event_time

        # Check traffic light logic and process queues
        if current_time >= next_light_switch:
            # Switch light group
            light_group = "east_west" if light_group == "north_south" else "north_south"
            next_light_switch = current_time + LIGHT_CYCLE_DURATION

        # Process vehicles in the active queue
        if queues[light_group]:
            lane, id = queues[light_group].popleft()
            start_time = max(current_time,queue_leave_times.get(id,current_time))
            processing_time = start_time + VEHICLE_PROCESSING_TIME

            queue_leave_times[id] = current_time
            processed_times[id] = processing_time

            current_time = processing_time  # Update the current time to the vehicle's departure time
        else:
            # No vehicles in the current group; skip to the next light switch or next arrival
            if arrival_events:
                next_event_time = arrival_events[0][AT]
                current_time = min(next_event_time,next_light_switch)
            else:
                current_time = next_light_switch

    return queue_arrival_times,queue_leave_times,processed_times


def main():
    # Read in the intersection lane data
    intersections = read_intersections() # This reads our manually create json file to get the lane data
    intersection_name = list(intersections.keys())[0] # Only get one intersection
    intersection_data = intersections[intersection_name] # Get the intersection data for that one intersection

    # set up lanes

    approaches=[] # n,e,s,w
    approaches.append([Queue() for x in range(intersection_data['north']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['east']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['south']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['west']['inlanes'])])

    # Read in the TomTom data for the intersections
    data = read_tomtom_data(date_times=["2024-12-01H21M00"], intersections=[OAKFOUL_VIC]) # Reads only for one date
    data_flat = flatten(data)

    # get traffic for this coordinate

    coords = intersection_data['coords']

    # Gather args for intersections in TomTom data
    ways=[]
    args_tomtom = {}
    for entry in data_flat:
        traffic_light_coord = entry['traffic_light_coord']
        if (traffic_light_coord["lat"] == str(coords["lat"]) and traffic_light_coord["long"] == str(coords["long"])):
            ways.append(entry)
            args_tomtom['current_speed'] = entry['currentSpeed']
            args_tomtom['free_flow_speed'] = entry["freeFlowSpeed"]
            args_tomtom['current_travel_time'] = entry["currentTravelTime"]
            args_tomtom['free_flow_travel_time'] = entry["freeFlowTravelTime"]

    num_inlanes = 0

    # Get number of in lanes
    for approach in approaches:
        for inlane in approach:
            num_inlanes += 1

    # Calculate lambda for the whole system
    lambda_free_flow = args_tomtom['free_flow_speed'] / (3.6 * (DIST_BETWEEN_CARS + AVG_CAR_LENGTH)) # s_Free / (3.6 * (c + L_car))
    congestion_factor = args_tomtom['free_flow_speed'] / args_tomtom['current_speed'] # s_Free / s_Current

    lambda_system = num_inlanes * lambda_free_flow * congestion_factor # lambda = N_{In Lanes} * lambda_{Free Flow per Lane} * R

    # Calculate lambda for each in lane
    lambdas = []

    # seperate lambda per queue/inlane, per approach
    for approach in approaches:
        dir_lambdas = []
        for inlane in approach:
            dir_lambdas.append(lambda_system / num_inlanes)
        lambdas.append(dir_lambdas)

    # simulate lanes
    num_events = 1000

    # fill inter-arrival times for approaches->lanes->IAT
    simulate_arrivals(approaches, lambdas, num_events)

    # merge queues into one giant queue (with queue as an attribute)    
    merged_arrival_times = merge_queues_with_priority(approaches) # intersection

    # simulate fixed-light intersection 
    queue_arrival_times,queue_departure_times,intersection_processed_times = simulate_intersection_fixed(merged_arrival_times)
    # values are epoch-based (not 'inter-...')

    # Print the results
    for i in queue_arrival_times.keys():
        QAT = round(queue_arrival_times[i],2) # time of entering queue
        QDT = round(queue_departure_times[i],2) # time of leaving queue / entering intersection
        IPT = round(intersection_processed_times[i],2) # time of leaving intersection
        QWT = round(QDT-QAT,2) # queue wait time
        IWT = round(IPT-QDT,2) # intersection wait time

        # only show first 20
        if i<20:
            print(i, 'QAT', QAT, 'QWT:', QWT, 'IWT:', IWT)

if __name__ == '__main__':
    main()