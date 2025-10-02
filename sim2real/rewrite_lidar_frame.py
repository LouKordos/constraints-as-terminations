import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class FrameRelabelNode(Node):
    def __init__(self):
        super().__init__('frame_relabel_node')
        # Subscribe to the original topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/utlidar/cloud',
            self.listener_callback,
            10)
        # Publisher for relabeled messages
        self.publisher = self.create_publisher(PointCloud2, '/pointcloud', 10)
        # Parameter for the target frame (default: 'base_link')
        self.declare_parameter('target_frame', 'radar')

    def listener_callback(self, msg: PointCloud2):
        # Clone the message header and set the new frame
        msg.header.frame_id = self.get_parameter('target_frame').get_parameter_value().string_value
        # Republish without altering point data
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FrameRelabelNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()




