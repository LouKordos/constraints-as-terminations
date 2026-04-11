import sys
import rosbag2_py
from datetime import datetime

def analyze_dt(bag_path, topic_name):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Failed to open bag. Ensure the rosbag2-storage-mcap plugin is installed.\nError: {e}")
        return

    storage_filter = rosbag2_py.StorageFilter(topics=[topic_name])
    reader.set_filter(storage_filter)
    timestamps = []
    print(f"Reading timestamps for {topic_name}...")
    
    while reader.has_next():
        topic, data, t = reader.read_next()
        timestamps.append(t) # 't' is recorded time in nanoseconds

    if len(timestamps) < 2:
        print("Error: Not enough messages recorded to calculate dt.")
        return

    start_time_ns = timestamps[0]
    dts = [(timestamps[i] - timestamps[i-1]) / 1e9 for i in range(1, len(timestamps))]
    avg_dt = sum(dts) / len(dts)
    max_dt = max(dts)
    
    hang_threshold = avg_dt * 3 # Define a spike as anything taking 3x longer than average (adjustable)
    
    spikes = [] # Track specific spikes for later printing
    for i in range(1, len(timestamps)):
        dt = (timestamps[i] - timestamps[i-1]) / 1e9
        if dt > hang_threshold:
            abs_time_ns = timestamps[i]
            rel_time_s = (abs_time_ns - start_time_ns) / 1e9
            readable_time = datetime.fromtimestamp(abs_time_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            spikes.append({
                'dt': dt,
                'relative_s': rel_time_s,
                'absolute_str': readable_time
            })

    print(f"\n--- DT Stats for {topic_name} ---")
    print(f"Total Messages: {len(timestamps)}")
    print(f"Min dt:         {min(dts):.4f} seconds")
    print(f"Max dt:         {max_dt:.4f} seconds")
    print(f"Average dt:     {avg_dt:.4f} seconds")
    print(f"---------------------------------")
    
    if not spikes:
        print(f"Data is clean, no spikes found where dt > {hang_threshold:.2f}s")
    else:
        print(f"Potential Hangs: Found {len(spikes)} instances where dt > {hang_threshold:.2f}s\n")
        print(f"{'Gap Duration':<15} | {'Relative Time (from start)':<30} | {'Absolute Timestamp'}")
        print("-" * 75)
        for s in spikes:
            print(f"{s['dt']:.4f}s{'':<7} | +{s['relative_s']:.3f} seconds{'':<11} | {s['absolute_str']}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 check_dt.py <path_to_bag_directory_or_file> <topic_name>")
        sys.exit(1)
    
    analyze_dt(sys.argv[1], sys.argv[2])
