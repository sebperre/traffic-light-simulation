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

class Lane:
    def __init__(self):
        pass
class Approach:
    def __init__(self, lanes:List[Lane]):
        self.lanes=lanes
class Intersection:
    def __init__(self, approaches:List[Approach]):
        self.approaches=approaches

def read_tomtom_data(num_times=100) -> dict:
    '''
    reads tomtom data into data dict object and adds traffic light coordinate
    '''
    cwd = os.getcwd()
    if not os.path.isdir('./cache/TOMTOM'):
        print('couldnt access ./cache/TOMTOM!')
        exit(1)
    os.chdir('./cache/TOMTOM')
    data = {}
    i = 0
    for dir in os.listdir('.'):
        if i < num_times:
            if os.path.isdir(dir):
                os.chdir(dir)
                if dir not in data.keys():
                    data[dir] = {}
                files = os.listdir('.')
                for file in files: # also date
                    if os.path.isfile(file):
                        with open(file,'r') as f:
                            contents = f.read()
                            if len(contents)==0:
                                continue
                            try:
                                entry = json.loads(contents) # also coords
                                r = re.findall(r'(-?\d+\.\d+)', file)
                                entry['flowSegmentData']['traffic_light_coord'] = {'lat': r[0], 'long': r[1]}
                            except:
                                print('parse fail', contents)
                                exit(1)
                            if file not in data[dir].keys():
                                data[dir][file] = entry['flowSegmentData']
                            else:
                                print(f"WARN: got duplicate data->'${dir}'->'${file}'")
            os.chdir('..')
            i += 1
        else:
            break
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
    data = read_tomtom_data(4)
    data_flat = flatten(data)

    intersections = read_intersections()
    intersection_name = list(intersections.keys())[0] # only the first
    intersection_data = intersections[intersection_name]

    # set up lanes

    approaches=[] # n,e,s,w
    approaches.append([Queue() for x in range(intersection_data['north']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['east']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['south']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['west']['inlanes'])])

    # get traffic for this coordinate

    coords = intersection_data['coords']

    ways=[]
    for entry in data_flat:
        traffic_light_coord = entry['traffic_light_coord']
        if (traffic_light_coord["lat"] == str(coords["lat"]) and traffic_light_coord["long"] == str(coords["long"])):
            ways.append(entry)
  
    # simulate lanes

    num_events = 1000
    lambdas=[ # seperate lambda per queue/inlane, per approach
        [1,2], # n
        [2,3], # e
        [3,4], # s
        [4,5]  # w
    ] 

    # Since we don't have lambda per lane and instead over the entire intersection,
    # we can simply band-aid this to put lambda/8 for each lambda
    lambda_ = 5

    # sum up number of lanes
    num_lanes = 0 
    for i in range(len(lambdas)):
        for j in range(len(lambdas[i])):
            num_lanes+=1

    # set lambda/num_lanes for each lane
    for i,lambs_in_approach in enumerate(lambdas): # approach
        for j,lam_lane in enumerate(lambs_in_approach): # lanes
            lambdas[i][j] = lambda_/num_lanes

    # fill inter-arrival-times for approaches->lanes->IAT
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