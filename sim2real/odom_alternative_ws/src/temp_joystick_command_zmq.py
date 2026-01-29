#!/usr/bin/env python3
import zmq
import time
import struct
import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState

class ZmqBridge(Node):
    def __init__(self):
        super().__init__("zmq_bridge_node")
        PORT = 6969

        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{PORT}")

        self.subscription = self.create_subscription(LowState, "/lowstate", self.listener_callback, 10)
        self.get_logger().info(f"ZMQ Joystick bridge started, forwarding /lowstate.wireless_remote to port {PORT}")

    
    def listener_callback(self, msg):
        data = msg.wireless_remote

        if len(data) < 24:
            self.get_logger().error("JOYSTICK DATA LENGTH TOO SHORT! Exiting")
            exit()
        # print(joystick_data)
        lx = struct.unpack('<f', bytes(data[4:8]))[0]
        rx = struct.unpack('<f', bytes(data[8:12]))[0]
        ry = struct.unpack('<f', bytes(data[12:16]))[0]
        ly = struct.unpack('<f', bytes(data[20:24]))[0] # Note: Offset 16-20 is skipped (L2 placeholder)

        btn_data = data[3]
        dpad_up    = (btn_data >> 4) & 1
        dpad_right = (btn_data >> 5) & 1
        dpad_down  = (btn_data >> 6) & 1
        dpad_left  = (btn_data >> 7) & 1

        # print(f"[Sticks] Ly: {ly:+.4f} | Lx: {lx:+.4f} | Ry: {ry:+.4f} | Rx: {rx:+.4f}  ||  [D-Pad] U:{dpad_up} D:{dpad_down} L:{dpad_left} R:{dpad_right}")

        try:
            self.socket.send(bytes(data))
            time.sleep(0.01)
        except Exception as ex:
            self.get_logger().error(f"Failed to serialize and send ZMQ message with joystick_data={joystick_data}. Exception: {ex}")


def main(args=None):
    rclpy.init(args=args)
    node = ZmqBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
