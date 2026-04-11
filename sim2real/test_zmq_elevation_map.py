#!/usr/bin/env python3
import zmq
import numpy as np
import sys

# Configuration matches your ElevationToPolicyNode
TARGET_IP = "192.168.123.224"
TARGET_PORT = 6973
GRID_WIDTH = 13  # x_points
GRID_HEIGHT = 11 # y_points
EXPECTED_FLOATS = 2 + GRID_WIDTH * GRID_HEIGHT # 145, 2 for date timestamp

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    print(f"Attempting to connect to tcp://{TARGET_IP}:{TARGET_PORT}...")
    
    # 1. Connect to the publisher
    try:
        socket.connect(f"tcp://{TARGET_IP}:{TARGET_PORT}")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    # 2. IMPORTANT: Subscribe to all messages
    # Without this, the SUB socket drops everything by default
    socket.setsockopt(zmq.SUBSCRIBE, b"")

    print(f"Connected. Listening for 'min_filter' (relative) layer data...")

    try:
        while True:
            # 3. Receive raw bytes
            # This blocks until data is available
            message = socket.recv()
            
            # 4. Decode binary data
            # The publisher sends: heights.tobytes() (float32 array)
            data = np.frombuffer(message, dtype=np.float32)

            # 5. Validation and Print
            print("-" * 40)
            print(f"Received Packet Size: {len(message)} bytes")
            
            if data.size == EXPECTED_FLOATS:
                # Calculate basic stats to ensure data looks real (not just zeros)
                min_val = np.min(data)
                max_val = np.max(data)
                mean_val = np.mean(data)
                
                print(f"Status: VALID FRAME")
                print(f"Shape:  {data.shape} (Expected {EXPECTED_FLOATS})")
                print(f"Values: Min: {min_val:.4f} | Max: {max_val:.4f} | Mean: {mean_val:.4f}")
                print(f"First 5 values: {data[:5]}")
            else:
                print(f"WARNING: Data size mismatch! Received {data.size} floats, expected {EXPECTED_FLOATS}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
