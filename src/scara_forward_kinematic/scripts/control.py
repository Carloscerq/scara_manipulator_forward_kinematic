#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_srvs.srv import Empty
import math
import random
import numpy as np
import tf2_ros
from tf.transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation

class ControlKinematic:

    pose_subscriber: rospy.Subscriber = None
    response_data: Pose = None
    joint_one_publisher: rospy.Publisher = None
    joint_two_publisher: rospy.Publisher = None
    joint_three_publisher: rospy.Publisher = None
    compare : bool = False

    d4: float = 1.0
    theta1: float = 0.0
    theta2: float = 0.0

    points = []
    truth = []
    pred = []

    angles_truth = []
    angles_pred = []

    def __init__(self):
        rospy.init_node('control_kinematic', anonymous=True)
        self.joint_one_publisher = rospy.Publisher("/scara_robot/joint1_position_controller/command", Float64, queue_size=10)
        self.joint_two_publisher = rospy.Publisher("/scara_robot/joint2_position_controller/command", Float64, queue_size=10)
        self.joint_three_publisher = rospy.Publisher("/scara_robot/joint3_position_controller/command", Float64, queue_size=10)
        self.pose_subscriber = rospy.Subscriber("/scara_robot/output_pose", Pose, self.pose_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.a = [0, 1, 1, 0]
        self.alpha = [0, 0, math.pi, 0]
        self.d = [0.3, 1, 0, self.d4]
        self.theta = [0, self.theta1, self.theta2, 0]

        self.compare = False
        self.pose_received = False  # <-- Added flag

        rospy.loginfo("Control Kinematic Node Initialized")

    def pose_callback(self, data: Pose):
        if not self.compare:
            return

        self.response_data = data
        trans = self.tf_buffer.lookup_transform("base", "link3", rospy.Time(), rospy.Duration(1.0))
        rospy.loginfo(f"Received Pose: {data}")
        rospy.loginfo(f"Received Pose: {trans}")
        self.compare = False
        self.pose_received = True  # <-- Flag set here

        # Calculate the A matrices for the forward kinematics
        A = []
        for i in range(4):
            A_i = np.array([
                [math.cos(self.theta[i]), -math.sin(self.theta[i]) * math.cos(self.alpha[i]), math.sin(self.theta[i]) * math.sin(self.alpha[i]), self.a[i] * math.cos(self.theta[i])],
                [math.sin(self.theta[i]), math.cos(self.theta[i]) * math.cos(self.alpha[i]), -math.cos(self.theta[i]) * math.sin(self.alpha[i]), self.a[i] * math.sin(self.theta[i])],
                [0, math.sin(self.alpha[i]), math.cos(self.alpha[i]), self.d[i]],
                [0, 0, 0, 1]
            ])
            A.append(A_i)

        T = np.eye(4, 4)
        for i in A:
            rospy.loginfo(f"A Matrix: {i}")
            T = np.dot(T, i)

        rospy.loginfo(f"Transformation Matrix T: {T}")
        end_effector_position = T[:3, 3]
        #end_effector_position[2] = data.position.z  # Set the z-coordinate to match the received pose
        rospy.loginfo(f"End Effector Position: {end_effector_position}")

        # Get the quaternion from the transform
        q = trans.transform.rotation
        quaternion = [q.x, q.y, q.z, q.w]

        # Convert to roll, pitch, yaw
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        rospy.loginfo(f"Truth Roll Pitch Yaw - {roll}, {pitch}, {yaw}")
        self.angles_truth.append([roll, pitch, yaw])

        rotation_matrix = T[:3, :3]
        rot = Rotation.from_dcm(rotation_matrix)
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
        rospy.loginfo(f"Pred Roll Pitch Yaw - {roll}, {pitch}, {yaw}")
        self.angles_pred.append([roll, pitch, yaw])

        self.pred.append(end_effector_position)
        self.truth.append([trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z])

    def run(self):
        rospy.loginfo("Control Kinematic Node Running")
        joint_one_value = Float64()
        joint_two_value = Float64()
        joint_three_value = Float64()
        joint_one_value.data = random.uniform(-1.0, 1.0)
        joint_two_value.data = random.uniform(-1.0, 1.0)
        joint_three_value.data = random.uniform(0.0, 1.0)
        self.joint_one_publisher.publish(joint_one_value)
        self.joint_two_publisher.publish(joint_two_value)
        self.joint_three_publisher.publish(joint_three_value)
        rospy.sleep(30)  # Wait for the initial pose to be received

        for i in range(11):
            joint_one_value = Float64()
            joint_two_value = Float64()
            joint_three_value = Float64()

            self.theta1 = random.uniform(-1.0, 1.0)
            self.theta2 = random.uniform(-1.0, 1.0)
            self.d4 = random.uniform(0, 1.0)

            joint_one_value.data = self.theta1
            joint_two_value.data = self.theta2
            joint_three_value.data = self.d4

            rospy.loginfo(f"Publishing Joint Values: Joint 1: {joint_one_value.data}, Joint 2: {joint_two_value.data}, Joint 3: {joint_three_value.data}")
            self.joint_one_publisher.publish(joint_one_value)
            self.joint_two_publisher.publish(joint_two_value)
            self.joint_three_publisher.publish(joint_three_value)

            self.theta[1] = self.theta1
            self.theta[2] = self.theta2
            self.d[3] = self.d4
            self.points.append([self.theta1, self.theta2, self.d4])

            rospy.sleep(10)

            #self.pose_callback()

            self.compare = True
            self.pose_received = False  # <-- Reset before waiting

            while not self.pose_received:
                rospy.loginfo("Waiting for pose callback to process the new joint values...")
                rospy.sleep(1)

        rospy.loginfo("Control Kinematic Node Finished Running")
        print("Points:")
        for i in self.points:
            print(i)

        print("pred")
        for i in self.pred:
            print(i)

        print("truth")
        for i in self.truth:
            print(i)

        print("angles - truth")
        for i in self.angles_truth:
            print(i)

        print("angles - pred")
        for i in self.angles_pred:
            print(i)


if __name__ == "__main__":
    try:
        control_kinematic = ControlKinematic()
        control_kinematic.run()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        rospy.loginfo("Control Kinematic Node Terminated")
