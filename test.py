def simulate_intersection_fixed(merged_arrival_times, vehicle_processing_rate: float = 2, light_cycle_duration: float = 10):
    # Changed here
    '''
    fixed interval light intersection
    logic: alternate between north-south, east-west queues.
    '''
    AT = 0  # arrival time index
    LANE = 1
    ID = 2

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
    next_light_switch = light_cycle_duration

    max_queue_len=0

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

        if len(queues[group])>max_queue_len:
            max_queue_len=len(queues[group])


        # Check traffic light logic and process queues
        if current_time >= next_light_switch:
            # Switch light group
            light_group = "east_west" if light_group == "north_south" else "north_south"
            next_light_switch = current_time + light_cycle_duration

        # Process vehicles in the active queue
        if queues[light_group]:
            lane, id = queues[light_group].popleft()
            start_time = max(current_time,queue_leave_times.get(id,current_time))
            vehicle_processing_time = np.random.exponential(1 / vehicle_processing_rate, 1)
            processing_time = start_time + vehicle_processing_time

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

    return max_queue_len,queue_arrival_times,queue_leave_times,processed_times