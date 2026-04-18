# header.stamp is the published time of the *processed* map, NOT the original elevation map. See source_pose_stamp for that.
# header.frame_id is global fixed frame in which the source GridMap and robot pose used for sampling are expressed (e.g. "odom", "map", "vicon/world").
# NOTE: The data array itself is NOT indexed in this frame, it is indexed in the yaw-aligned local policy grid defined by size_x, size_y, resolution, and sensor_offset_x.
std_msgs/Header header

# Sequence ID. Increments by 1 each time the perception node publishes.
uint64 seq

# Original timestamp of the raw elevation map
builtin_interfaces/Time source_map_stamp
# Timestamp of the robot pose / TF lookup used to generate the yaw-aligned local policy observation. Different from source_map_stamp because the two run concurrently
builtin_interfaces/Time source_pose_stamp

uint32 size_x           # Number of cells in the X direction
uint32 size_y           # Number of cells in the Y direction
float32 resolution      # Size of each cell in meters/cell
float32 sensor_offset_x # Because simulation was trained with offset in x dir

# Robot pose in header.frame_id (global frame)
geometry_msgs/Pose robot_pose

# Handling of invalid/occluded cells
float32 fill_value      # CANNOT BE NaN, as values are assumed to sanitized for being passed directly to policy inference
uint8[] is_valid         # boolean mask, but not bool type because of std::vector<bool> concerns, so 1 = valid, 0 = invalid

# Name of the source layer in the incoming GridMap, e.g. "elevation", "min_filter", "smooth".
string layer_name

# Actual map observations, row-major order, and contains sampled height values computed as z_world_sample - z_world_base,
# where XY sample locations are generated from a yaw-only rotation of the local query grid around the robot base pose.
# To retrieve absolute global elevation, add the robot's current global Z position to these values.
# data[row * size_x + col] corresponds to
# x = (col - (size_x - 1)/2) * resolution + sensor_offset_x
# y = (row - (size_y - 1)/2) * resolution
# in the yaw-aligned local frame used by the policy.
float32[] data