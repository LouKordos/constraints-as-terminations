import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import zmq
import time

class CmdVelSubscriberZMQ(Node):
    def __init__(self):
        super().__init__('cmd_vel_zmq_translation')

        # Declare parameters
        self.declare_parameter('ip', '127.0.0.1')
        self.declare_parameter('port', 6969)

        # Retrieve parameters
        zmq_ip = self.get_parameter('ip').get_parameter_value().string_value
        zmq_port = self.get_parameter('port').get_parameter_value().integer_value
        zmq_endpoint = f"tcp://{zmq_ip}:{zmq_port}"

        # ZMQ setup
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect(zmq_endpoint)
        self.get_logger().info(f"Connected to ZMQ endpoint: {zmq_endpoint}")

        # Subscribe to /cmd_vel
        self.create_subscription(
            TwistStamped,
            '/cmd_vel',
            self.cmd_vel_callback,
            10  # QoS history depth
        )

    def cmd_vel_callback(self, msg: TwistStamped):
        ts_ms = int(time.time() * 1000)
        vel_x = msg.twist.linear.x
        vel_y = msg.twist.linear.y
        ang_z = msg.twist.angular.z
        payload = f"{ts_ms},{vel_x},{vel_y},{ang_z}"
        try:
            self.socket.send_string(payload)
            self.get_logger().debug(f"Sent: {payload}")
        except zmq.ZMQError as e:
            self.get_logger().error(f"ZMQ send error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSubscriberZMQ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down subscriber')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
