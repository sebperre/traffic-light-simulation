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
import random
from scipy.stats import t,norm
import math

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

KM_H_TO_M_S=3.6

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

    rows = len(queues)
    cols = len(queues[0]) if rows > 0 else 0

    inter_arrival_times = [[[] for _ in range(cols)] for _ in range(rows)]

    for num_event in range(num_events):
        rand_queue_index = random.choice(range(len(queues)))
        assert(len(queues[rand_queue_index]) == len(lambdas[rand_queue_index]))
        rand_inlane_index = random.choice(range(len(queues[rand_queue_index])))

        lam = lambdas[rand_queue_index][rand_inlane_index]
        inter_arrival_time = random.expovariate(lam)
        inter_arrival_times[rand_queue_index][rand_inlane_index].append(inter_arrival_time)
    
    for i, direction in enumerate(queues):
        for j, inlane in enumerate(direction):
            arrival_times = np.cumsum(inter_arrival_times[i][j])
            queues[i][j]=arrival_times # overwrite lane as it should be empty at this point

def merge_queues_with_priority(approaches): 
    '''
    merge approaches->lanes->... into one big PQ
    '''
    merged_queue = []
    lane_count=0
    for i,approach in enumerate(approaches):
        for j,lane in enumerate(approach):
            for arrival_time in lane:
                heapq.heappush(merged_queue, (arrival_time, i, lane_count))  # (time, (approach, lane))
            lane_count+=1
    return merged_queue

def simulate_intersection_fixed(approaches, merged_arrival_times, free_flow_speed, current_speed, intersection_length, light_cycle_duration: float = 10):
    '''
    fixed interval light intersection, on light_cycle_duration intervals.
    alternates between north-south, east-west.
    assumes that there are 4 approaches; n,e,s,w
    intersection delay/processing time is determined by current_speed and distance.
    '''
    # tuple index constants
    AT = 0
    LANE = 1 
    ID = 2

    # group constants (we know there will always be 4 groups)
    NORTH_SOUTH = (0,2)
    EAST_WEST = (1,3)

    def have_active_queues():
        # returns True if we have at least one lane with something in it.
        # False otherwise.
        for lane in queues:
            if len(lane)>0:
                return True
        return False

    def get_group_queues(group):
        qs=[]
        ids=[]
        for g in group:
            for q in lane_groups[g]:
                ids.append(q)
                qs.append(queues[q])
        return qs,ids

    def queues_active(queue_ids):
        active_queues=[]
        active_ids=[]
        for id in queue_ids:
            if queues[id]:
                active_queues.append(queues[id])
                active_ids.append(id)
        return active_queues,active_ids

    current_time = 0
    arrival_events = []
    queues = [] # 0-7 for 8 lanes
    lane_groups={} # group id->lane # (0-7 for 8 lanes)
    num_queuelanes=0

    # fill out queues and lane_groups
    for i,approach in enumerate(approaches):
        if i not in lane_groups.keys():
            lane_groups[i]=[]

        for lane in approach:
            queues.append(deque())
            lane_groups[i].append(num_queuelanes)
            num_queuelanes=num_queuelanes+1

    # push arriving vehicles onto heap, preserving what lane it is.
    num_vehicles = 0 # id
    for arrival_time, approach, lane in merged_arrival_times:
        heappush(arrival_events, (arrival_time, lane, num_vehicles)) # lane = group
        num_vehicles += 1

    # Traffic light management
    light_group = NORTH_SOUTH if random.random()>=0.5 else EAST_WEST
    next_light_switch = light_cycle_duration

    # metrics
    max_queue_len=0
    queue_arrival_times = {}
    queue_leave_times = {}
    processed_times = {}

    delay_start_from_stop = [random.uniform(1,4) for _ in range(num_queuelanes)] # delay from stop
    green_light_num_processed=0
    green_light_time_start=0
    green_light_time=0

    any_queue_is_active = have_active_queues()

    completeTimes=[]
    beginTime=None

    while arrival_events or any_queue_is_active:
        if not any_queue_is_active: # Advance time to the next event
            if arrival_events:
                current_time = max(current_time, arrival_events[0][AT])
            else:
                # No vehicles left to process
                break
        
        # Process arrival events
        while arrival_events and arrival_events[0][AT] <= current_time:
            event_time, lane, id = heappop(arrival_events)

            queues[lane].append(id)
            queue_arrival_times[id] = event_time

        # kep track of max queue length
        if len(queues[lane])>max_queue_len:
            max_queue_len=len(queues[lane])

        # Traffic light logic
        # simple time-based switch
        if current_time >= next_light_switch:
            light_group = EAST_WEST if light_group == NORTH_SOUTH else NORTH_SOUTH
            next_light_switch = current_time + light_cycle_duration 
            delay_start_from_stop = [random.uniform(1,4) for _ in range(num_queuelanes)] 

        # Process vehicles in light_group queues (opposing directions)
        queue_index = None
        _,group_ids = get_group_queues(light_group)
        active_queues,_ = queues_active(group_ids)

        if active_queues:
            for queue in active_queues:

                if current_time>=next_light_switch:
                    light_group = EAST_WEST if light_group == NORTH_SOUTH else NORTH_SOUTH
                    next_light_switch = current_time + light_cycle_duration 
                    delay_start_from_stop = [random.uniform(1,4) for _ in range(num_queuelanes)] 
                    break

                if queue: # has vehicles
                    id = queue.popleft()
                    # print('process',id)

                    beginTime=current_time
                    
                    queue_index = queues.index(queue)

                    assert(current_time==queue_leave_times.get(id, current_time))
                    
                    # time spent in intersection.
                    # t = d/v
                    if delay_start_from_stop[queue_index]==0:
                        t = intersection_length/free_flow_speed
                    else:
                        t = intersection_length/current_speed

                    # processing time includes time to start from stop AND time in intersection
                    processing_time = delay_start_from_stop[queue_index] + t
                    # print(processing_time)

                    queue_leave_times[id] = current_time

                    current_time = current_time + processing_time

                    processed_times[id] = current_time

                    # other cars won't be delayed any further, 
                    # as this car delays the others by setting curt+pt
                    delay_start_from_stop[queue_index] = 0

        else: # skip time to the next light switch

            if beginTime!=None:
                completeTimes.append(current_time-beginTime)
            beginTime=None

            if arrival_events:
                next_event_time = arrival_events[0][AT]
                current_time = min(next_event_time, next_light_switch)
            else:
                assert(next_light_switch>current_time) # if this fails, we're processing cars while we have a red light!
                current_time = next_light_switch
        any_queue_is_active = have_active_queues()

    if (len(completeTimes)>0):
        avg=np.mean(completeTimes)
    else:
        avg=0

    return current_time,max_queue_len,queue_arrival_times,queue_leave_times,processed_times,avg

def confidence_interval(data, alpha): # per minute
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)

    std_error = std_dev / np.sqrt(n)

    # z_score = norm.ppf(1-alpha/2) 
    t_score = t.ppf(1-alpha/2, n-1)

    lower_bound = mean - t_score * std_error
    upper_bound = mean + t_score * std_error

    return lower_bound,upper_bound

def run_simulation(date_time, intersection, intersection_coords, approaches, num_events):
    # Read in the TomTom data for the intersections
    data = read_tomtom_data(date_times=[date_time], intersections=[intersection]) # Reads only for one date
    data_flat = flatten(data)

    # Gather args for intersections in TomTom data
    ways=[]
    args_tomtom = {}
    for entry in data_flat:
        traffic_light_coord = entry['traffic_light_coord']
        if (traffic_light_coord["lat"] == str(intersection_coords["lat"]) and traffic_light_coord["long"] == str(intersection_coords["long"])):
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

    # Calculated visually
    intersection_length = 26.64 # meters

    # Calculate lambda for the whole system
    lambda_free_flow = args_tomtom['free_flow_speed'] / (KM_H_TO_M_S * (DIST_BETWEEN_CARS + AVG_CAR_LENGTH)) # s_Free / (3.6 * (c + L_car))
    congestion_factor = args_tomtom['free_flow_speed'] / args_tomtom['current_speed'] # s_Free / s_Current
    lambda_system = num_inlanes * lambda_free_flow * congestion_factor # lambda = N_{In Lanes} * lambda_{Free Flow per Lane} * R
    vehicle_processing_rate = args_tomtom['free_flow_speed'] / (KM_H_TO_M_S * intersection_length)

    # Calculate lambda for each in lane
    lambdas = []

    # seperate lambda per queue/inlane, per approach
    for approach in approaches:
        dir_lambdas = []
        for inlane in approach:
            dir_lambdas.append(lambda_system / num_inlanes)
        lambdas.append(dir_lambdas)

    # simulate lanes

    # fill inter-arrival times for approaches->lanes->IAT
    simulate_arrivals(approaches, lambdas, num_events)

    # merge queues into one giant queue (with queue as an attribute)    
    merged_arrival_times = merge_queues_with_priority(approaches) # intersection

    light_time=20

    # simulate fixed-light intersection 
    simulation_end_time,max_queue_len,queue_arrival_times,queue_departure_times,intersection_processed_times,avg_intersection_clear_time = simulate_intersection_fixed(approaches, merged_arrival_times, args_tomtom['free_flow_speed']/3.6, args_tomtom['current_speed']/3.6, intersection_length, light_time)

    waiting_times=[]
    avg_intersection_clear_time_avg=[]

    # Print the results
    for i in queue_arrival_times.keys():

        TimeEnterQueue = queue_arrival_times[i] # time of entering queue
        TimeEnterIntersection = queue_departure_times[i] # time of leaving queue / entering intersection
        TimeLeaveIntersection = intersection_processed_times[i] # time of leaving intersection

        QueueWaitTime = TimeEnterIntersection-TimeEnterQueue # queue wait time
        IntersectionProcessTime = TimeLeaveIntersection-TimeEnterIntersection # Intersection processing Time (Time for one vehicle to get from the beginning of the intersection to the end)
        TotalWaitTime = TimeLeaveIntersection-TimeEnterQueue

        avg_intersection_clear_time_avg.append(avg_intersection_clear_time)

        waiting_times.append(QueueWaitTime)

    events_per_second = num_events/simulation_end_time

    average_waiting_time=(sum(waiting_times)/len(waiting_times))

    return waiting_times,average_waiting_time,events_per_second,max_queue_len,avg_intersection_clear_time_avg

def main():
    # Read in the intersection lane data
    intersections = read_intersections() # This reads our manually create json file to get the lane data
    intersection_name = list(intersections.keys())[0] # Only get one intersection
    intersection_data = intersections[intersection_name] # Get the intersection data for that one intersection
    intersection_coords = intersection_data['coords']

    # set up lanes

    approaches=[] # n,e,s,w
    approaches.append([Queue() for x in range(intersection_data['north']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['east']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['south']['inlanes'])])
    approaches.append([Queue() for x in range(intersection_data['west']['inlanes'])])

    waiting_times_all=[]
    average_waiting_time_all=[]
    events_per_second_all=[]
    max_queue_len_all=[]
    throughputs=[]
    traffic_flow_efficiency_all=[]

    datetimes=["2024-12-01H09M00","2024-12-01H12M00","2024-12-01H15M00","2024-12-01H18M00","2024-12-01H21M00",]
    num_simulations=10
    num_vehicles=1000

    for date in datetimes:
        for i in range(num_simulations):
            waiting_times,average_waiting_time,events_per_second,max_queue_len,avg_intersection_clear_time = run_simulation(date, OAKFOUL_VIC, intersection_coords, approaches, num_vehicles)
            waiting_times_all.append(waiting_times)
            average_waiting_time_all.append(average_waiting_time)
            events_per_second_all.append(events_per_second)
            max_queue_len_all.append(max_queue_len)
            traffic_flow_efficiency_all.append(avg_intersection_clear_time)
            throughputs.append(events_per_second)
        
        throughputs_ci = confidence_interval(throughputs,0.05)
        low = throughputs_ci[0]
        high = throughputs_ci[1]

        throughputs_mean = np.mean(throughputs)

        print(f'Metrics for fixed light control, over {num_simulations} runs, at {date}')
        print('Average waiting time', round(sum(average_waiting_time_all)/len(average_waiting_time_all),4))
        print(f'Throughput/min CI @95% [{round(low*60,4)}, {round(high*60,4)}], mu={round(throughputs_mean*60,4)}')
        print('Max queue length', round(sum(max_queue_len_all)/len(max_queue_len_all),4))
        print('Traffic flow efficiency (avg time to clear intersection)', round(np.mean(traffic_flow_efficiency_all),4))
    
if __name__ == '__main__':
    main()