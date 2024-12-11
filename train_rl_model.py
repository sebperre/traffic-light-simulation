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

# def merge_queues_with_priority(approaches): 
#     '''
#     merge approaches->lanes->... into one big PQ
#     '''
#     merged_queue = []
#     for i,approach in enumerate(approaches):
#         for j,lane in enumerate(approach):
#             for arrival_time in lane:
#                 heapq.heappush(merged_queue, (arrival_time, i, j))  # (time, (approach, lane))
#     return merged_queue

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

    # Calculated visually
    intersection_length = 26.64 # meters

    # Calculate lambda for the whole system
    lambda_free_flow = args_tomtom['free_flow_speed'] / (3.6 * (DIST_BETWEEN_CARS + AVG_CAR_LENGTH)) # s_Free / (3.6 * (c + L_car))
    congestion_factor = args_tomtom['free_flow_speed'] / args_tomtom['current_speed'] # s_Free / s_Current
    lambda_system = num_inlanes * lambda_free_flow * congestion_factor # lambda = N_{In Lanes} * lambda_{Free Flow per Lane} * R
    vehicle_processing_rate = args_tomtom['free_flow_speed'] / (3.6 * intersection_length)

    # Calculate lambda for each in lane
    lambdas = []

    # seperate lambda per queue/inlane, per approach
    for approach in approaches:
        dir_lambdas = []
        for inlane in approach:
            dir_lambdas.append(lambda_system / num_inlanes)
        lambdas.append(dir_lambdas)

    # simulate lanes
    num_events = 20

    # fill inter-arrival times for approaches->lanes->IAT
    simulate_arrivals(approaches, lambdas, num_events)

    print(approaches)
    cars_per_direction = []
    for approach in approaches:
        count = 0
        for lane in approach:
            for el in lane:
                count += 1
        cars_per_direction.append(count)

    print(cars_per_direction)

    def get_cars_added_at_second(t: int, approaches) -> list:
        cars_per_direction_at_t = []
        for approach in approaches:
            count = 0
            for lane in approach:
                for el in lane:
                    if t-1 < el and el <= t:
                        count += 1
            cars_per_direction_at_t.append(count)
        return cars_per_direction_at_t

    

if __name__ == '__main__':
    main()


import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cpu")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# we will need to get the number of cars everysingle second
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space

# Actions are Turn North-South green and East-West red or North-South red and East-West green
n_actions = 2

# States are [North, num_lanes_north, cars_north, South, num_lanes_south, cars_south, east, num_lanes_east, num_cars_east, west, num_lanes_west, cars_west, east, num_lanes_east, cars_east]
n_observations = 12

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Update states every second

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()