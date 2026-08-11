#!/usr/bin/env python3
import argparse
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

class TopicWaiter(Node):
    def __init__(self, topics, timeout_sec):
        super().__init__("wait_for_topics")
        self._remaining_topics = set(topics)
        self._topic_subscriptions = {}
        self._timeout_sec = timeout_sec
        self._start_time = time.monotonic()
        self._error_message = None

    @property
    def error_message(self):
        return self._error_message

    @property
    def all_topics_received(self):
        return not self._remaining_topics

    @property
    def timed_out(self):
        return self._timeout_sec > 0.0 and (time.monotonic() - self._start_time) > self._timeout_sec

    @property
    def remaining_topics(self):
        return sorted(self._remaining_topics)

    def discover_and_subscribe(self):
        topic_names_and_types = dict(self.get_topic_names_and_types())
        for topic in list(self._remaining_topics):
            if topic in self._topic_subscriptions:
                continue

            topic_types = topic_names_and_types.get(topic)
            if not topic_types:
                continue

            if len(topic_types) != 1:
                self._error_message = (
                    f"Topic '{topic}' has multiple types {topic_types}. "
                    "This helper only supports topics with exactly one type."
                )
                return

            topic_type = topic_types[0]
            try:
                message_type = get_message(topic_type)
            except (AttributeError, ModuleNotFoundError, ValueError) as exception:
                self._error_message = (
                    f"Failed to import message type '{topic_type}' for topic '{topic}': {exception}"
                )
                return

            self._topic_subscriptions[topic] = self.create_subscription(
                message_type,
                topic,
                self._make_callback(topic),
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f"Subscribed to '{topic}' with type '{topic_type}', waiting for first message..."
            )

    def _make_callback(self, topic):
        def callback(_message):
            if topic not in self._remaining_topics:
                return

            self._remaining_topics.remove(topic)
            self.get_logger().info(
                f"Received first message on '{topic}'. Remaining topics: {self.remaining_topics}"
            )
        return callback

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wait until at least one message has been received on each requested topic."
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Maximum time to wait before exiting with a failure code. Use 0 or a negative value to wait indefinitely.",
    )
    parser.add_argument(
        "topics",
        nargs="+",
        help="Topics that must each produce at least one message.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    topics = list(dict.fromkeys(args.topics))
    rclpy.init()
    node = TopicWaiter(topics=topics, timeout_sec=args.timeout_sec)
    exit_code = 1

    try:
        node.get_logger().info(f"Waiting for first message on topics: {topics}")
        while rclpy.ok():
            node.discover_and_subscribe()
            if node.error_message is not None:
                node.get_logger().error(node.error_message)
                exit_code = 1
                break

            if node.all_topics_received:
                node.get_logger().info("Received at least one message on all requested topics.")
                exit_code = 0
                break

            if node.timed_out:
                node.get_logger().error(
                    f"Timed out after {args.timeout_sec:.1f} seconds while waiting for topics: {node.remaining_topics}"
                )
                exit_code = 1
                break
            rclpy.spin_once(node, timeout_sec=0.2)

    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()