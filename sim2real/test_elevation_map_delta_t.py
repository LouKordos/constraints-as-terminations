import zmq
import time
import sys
import json

def monitor_min_filter_layer_deep():
    endpoint = "tcp://192.168.123.224:6973"
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVHWM, 2)

    try:
        print(f"Connecting to {endpoint}...")
        socket.connect(endpoint)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"Subscribed to endpoint={endpoint}. Monitoring true age and dropped frames...\n")
        
        last_arrival_time = None
        last_payload_ts = None
        
        while True:
            msg = socket.recv()  
            current_time = time.perf_counter()
            
            try:
                payload = json.loads(msg.decode('utf-8'))
                current_payload_ts = payload.get("ts")
            except Exception as e:
                print(f"JSON Parse Error: {e}")
                continue
            
            if last_arrival_time is not None and last_payload_ts is not None:
                arrival_delta_ms = (current_time - last_arrival_time) * 1000.0
                
                # Compare the timestamps applied by the ROS node. Multiplying by 1000 because ROS ts is in seconds
                published_delta_ms = (current_payload_ts - last_payload_ts) * 1000.0
                
                if arrival_delta_ms > 30.0:
                    print("-" * 50)
                    print("DELAY SPIKE DETECTED")
                    print(f"  Time since last receive: {arrival_delta_ms:.1f} ms")
                    print(f"  Gap between ROS stamps:  {published_delta_ms:.1f} ms")
            
            last_arrival_time = current_time
            last_payload_ts = current_payload_ts

    except KeyboardInterrupt:
        print("\nShutting down monitor...")
    finally:
        socket.close()
        context.term()
        sys.exit(0)

if __name__ == "__main__":
    monitor_min_filter_layer_deep()



