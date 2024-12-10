import os
import json
import re
from typing import List
from decimal import Decimal, ROUND_DOWN
from queue import Queue
from math import radians, sin, cos, sqrt, atan2

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
    cwd=os.getcwd()
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

def main():
    data = read_tomtom_data(4)
    data_flat = flatten(data)

    intersections = read_intersections()
    intersection_name = list(intersections.keys())[0] # only the first
    intersection_data = intersections[intersection_name]

    # set up lanes

    lanes={}
    for i in range(intersection_data['north']['inlanes']):
        if 'north' not in lanes:
            lanes['north']=[]
        lanes['north'].append(Queue())
    for i in range(intersection_data['east']['inlanes']):
        if 'east' not in lanes:
            lanes['east']=[]
        lanes['east'].append(Queue())
    for i in range(intersection_data['south']['inlanes']):
        if 'south' not in lanes:
            lanes['south']=[]
        lanes['south'].append(Queue())
    for i in range(intersection_data['west']['inlanes']):
        if 'west' not in lanes:
            lanes['west']=[]
        lanes['west'].append(Queue())
    print(lanes)

    # get traffic for this coordinate

    coords = intersection_data['coords']

    ways=[]
    for entry in data_flat:
        traffic_light_coord = entry['traffic_light_coord']
        if (traffic_light_coord["lat"] == str(coords["lat"]) and traffic_light_coord["long"] == str(coords["long"])):
            ways.append(entry)
    print(len(ways))

if __name__ == '__main__':
    main()