import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

# Control period [s]
TIMER = 0.1

# Image coordinate of the vehicle center used in path_planner_node.py
CAR_CENTER_X = 320.0

# PID gains for lateral pixel error
# Start from these values and tune in simulation.
KP = 0.05
KI = 0.00
KD = 0.01

# PID / steering stabilization
INTEGRAL_LIMIT = 300.0
DERIVATIVE_FILTER_ALPHA = 0.75
ERROR_DEADBAND_PX = 3.0
MAX_STEERING = 7.0
MAX_STEERING_STEP = 7.0

# path_data is ordered from far -> near.
# Average a band of points ahead instead of using one noisy point.
LOOKAHEAD_FAR_OFFSET = 100
LOOKAHEAD_NEAR_OFFSET = 50
# Lower speed first while tuning the steering controller.
DRIVE_SPEED = 255
#----------------------------------------------


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # Topic parameters
        self.sub_detection_topic = self.declare_parameter(
            'sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter(
            'sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter(
            'sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter(
            'sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = float(self.declare_parameter('timer', TIMER).value)

        # PID parameters can be tuned with ROS parameters if desired.
        self.kp = float(self.declare_parameter('kp', KP).value)
        self.ki = float(self.declare_parameter('ki', KI).value)
        self.kd = float(self.declare_parameter('kd', KD).value)
        self.drive_speed = int(self.declare_parameter('drive_speed', DRIVE_SPEED).value)

        # QoS
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Inputs
        self.detection_data = None
        self.path_data = None
        self.traffic_light_data = None
        self.lidar_data = None

        # Commands
        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # PID state
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.filtered_derivative = 0.0
        self.steering_float = 0.0

        # Subscribers
        self.detection_sub = self.create_subscription(
            DetectionArray, self.sub_detection_topic,
            self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(
            PathPlanningResult, self.sub_path_topic,
            self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(
            String, self.sub_traffic_light_topic,
            self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(
            Bool, self.sub_lidar_obstacle_topic,
            self.lidar_callback, self.qos_profile)

        # Publisher
        self.publisher = self.create_publisher(
            MotionCommand, self.pub_topic, self.qos_profile)

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def reset_pid(self):
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.filtered_derivative = 0.0
        self.steering_float = 0.0

    def get_lateral_error(self):
        """
        Return lateral path error in image pixels.

        path_data is far -> near because path_planner_node.py sorts by y and
        appends the vehicle center at the largest y.  We average several
        look-ahead points to suppress frame-to-frame perception jitter.

        Positive error: path center is to the right of the vehicle center.
        """
        if self.path_data is None or len(self.path_data) < 10:
            return None

        n = len(self.path_data)
        start = max(0, n - LOOKAHEAD_FAR_OFFSET)
        end = max(start + 1, n - LOOKAHEAD_NEAR_OFFSET)
        lookahead_points = self.path_data[start:end]

        if not lookahead_points:
            lookahead_points = [self.path_data[max(0, n - 30)]]

        target_x = sum(point[0] for point in lookahead_points) / len(lookahead_points)
        return target_x - CAR_CENTER_X

    def calculate_pid_steering(self):
        error = self.get_lateral_error()
        if error is None:
            self.reset_pid()
            return 0

        # Ignore very small pixel jitter around the lane center.
        if abs(error) < ERROR_DEADBAND_PX:
            error = 0.0

        dt = max(self.timer_period, 1e-3)

        # Integral with anti-windup.
        self.integral_error += error * dt
        self.integral_error = max(
            -INTEGRAL_LIMIT,
            min(INTEGRAL_LIMIT, self.integral_error)
        )

        # Derivative + low-pass filtering to avoid reacting to YOLO/path noise.
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (
            DERIVATIVE_FILTER_ALPHA * self.filtered_derivative
            + (1.0 - DERIVATIVE_FILTER_ALPHA) * raw_derivative
        )

        pid_output = (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * self.filtered_derivative
        )

        # Preserve the steering direction of the previous bang-bang controller.
        # A path center to the right produced a negative MotionCommand steering
        # in the existing pipeline because simulation_sender applies STEERING=-1.
        desired_steering = pid_output
        desired_steering = max(-MAX_STEERING, min(MAX_STEERING, desired_steering))

        # Steering slew-rate limiter: prevents +7 <-> -7 instant switching.
        delta = desired_steering - self.steering_float
        delta = max(-MAX_STEERING_STEP, min(MAX_STEERING_STEP, delta))
        self.steering_float += delta

        self.prev_error = error

        self.get_logger().info(
            f"lane_error={error:.1f}px, "
            f"P={self.kp * error:.2f}, "
            f"I={self.ki * self.integral_error:.2f}, "
            f"D={self.kd * self.filtered_derivative:.2f}, "
            f"steer={self.steering_float:.2f}"
        )

        return int(round(self.steering_float))

    def should_stop_for_red_light(self):
        if self.traffic_light_data is None:
            return False
        if self.traffic_light_data.data != 'Red':
            return False
        if self.detection_data is None:
            return False

        for detection in self.detection_data.detections:
            if detection.class_name != 'light':
                continue

            y_max = int(
                detection.bbox.center.position.y
                + detection.bbox.size.y / 2
            )

            if y_max < 150:
                return True

        return False

    def timer_callback(self):
        stop_for_lidar = (
            self.lidar_data is not None and self.lidar_data.data is True
        )
        stop_for_red = self.should_stop_for_red_light()

        if stop_for_lidar or stop_for_red:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            self.reset_pid()
        else:
            self.steering_command = self.calculate_pid_steering()
            self.left_speed_command = self.drive_speed
            self.right_speed_command = self.drive_speed

        self.get_logger().info(
            f"steering: {self.steering_command}, "
            f"left_speed: {self.left_speed_command}, "
            f"right_speed: {self.right_speed_command}"
        )

        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
