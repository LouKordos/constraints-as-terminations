#!/usr/bin/env python3
import zmq
import time
import argparse
import numpy as np
import sys
from collections import deque

class ZmqRateMonitor:
    """
    Monitors a ZeroMQ Publisher socket to calculate message frequency and jitter.
    Mimics the behavior of 'ros2 topic hz'.
    """
    def __init__(self, port: int, window_size: int = 100):
        """
        Initialize the monitor.

        Args:
            port: The TCP port to subscribe to.
            window_size: Number of messages to keep in history for statistical calculation.
        """
        self.port = port
        self.window_size = window_size
        self.zmq_context = zmq.Context()
        self.socket = None
        self.is_running = False
        
        # Buffer to store timestamps of message arrival
        self.arrival_times_buffer = deque(maxlen=window_size)

    def connect(self):
        """
        Establishes the ZMQ subscription.
        """
        try:
            self.socket = self.zmq_context.socket(zmq.SUB)
            connection_string = f"tcp://192.168.123.224:{self.port}"
            self.socket.connect(connection_string)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics
            print(f"Subscribed to ZMQ Publisher on {connection_string}")
            print(f"Collecting data (window size: {self.window_size})...")
        except zmq.ZMQError as e:
            print(f"Failed to connect to ZMQ socket: {e}")
            sys.exit(1)

    def run(self):
        """
        Main loop. Blocks execution to receive messages and prints statistics.
        """
        self.is_running = True
        self.connect()

        last_print_time = time.time()
        
        try:
            while self.is_running:
                # Block until a message is received
                # We don't need to deserialize the JSON to check the rate, 
                # just knowing bytes arrived is sufficient.
                _ = self.socket.recv()
                
                current_time = time.time()
                self.arrival_times_buffer.append(current_time)

                # Check if it is time to print statistics (approx every 1 second)
                if current_time - last_print_time >= 1.0:
                    self.print_statistics()
                    last_print_time = time.time()

        except KeyboardInterrupt:
            print("\nMonitor stopped by user.")
        except Exception as e:
            print(f"\nUnexpected error in monitor loop: {e}")
        finally:
            self.cleanup()

    def print_statistics(self):
        """
        Calculates and formats the frequency statistics based on the buffer.
        """
        if len(self.arrival_times_buffer) < 2:
            print("Waiting for more data...")
            return

        # Convert deque to numpy array for easier math
        timestamps = np.array(self.arrival_times_buffer)
        
        # Calculate differences between consecutive timestamps (inter-arrival times)
        inter_arrival_times = np.diff(timestamps)
        
        if len(inter_arrival_times) == 0:
            return

        # Frequency = 1 / mean_inter_arrival_time
        mean_interval = np.mean(inter_arrival_times)
        if mean_interval == 0:
            return

        rate_hz = 1.0 / mean_interval
        min_interval = np.min(inter_arrival_times)
        max_interval = np.max(inter_arrival_times)
        std_dev = np.std(inter_arrival_times)

        # formatted output similar to ros2 topic hz
        output = (
            f"Average Rate: {rate_hz:6.2f} Hz\n"
            f"    Min: {min_interval:.4f}s, Max: {max_interval:.4f}s, "
            f"StdDev: {std_dev:.4f}s, Window: {len(self.arrival_times_buffer)}"
        )
        print("-" * 40)
        print(output)

    def cleanup(self):
        """
        Cleanly closes sockets and contexts.
        """
        if self.socket:
            self.socket.close()
        self.zmq_context.term()

def get_port_from_args():
    """
    Parses command line arguments to determine the target port.
    """
    parser = argparse.ArgumentParser(description="Check rate of ZMQ Elevation Publisher.")
    
    # Mutually exclusive group: either specify port directly or use layer/type helper
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port", type=int, help="Direct TCP port number to listen to.")
    group.add_argument("--layer", type=str, choices=["elevation", "min_filter", "smooth"], 
                       help="Layer name to resolve port automatically.")
    
    parser.add_argument("--type", type=str, choices=["abs", "rel"], default="abs",
                        help="Data type if using --layer (default: abs).")
    
    args = parser.parse_args()

    if args.port:
        return args.port
    
    # Mapping must match the Publisher Node configuration
    base_ports = {
        "elevation": 6970,
        "min_filter": 6972,
        "smooth": 6974
    }
    
    selected_base_port = base_ports[args.layer]
    final_port = selected_base_port if args.type == "abs" else selected_base_port + 1
    return final_port

if __name__ == "__main__":
    target_port = get_port_from_args()
    monitor = ZmqRateMonitor(port=target_port)
    monitor.run()
