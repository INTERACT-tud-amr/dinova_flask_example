#!/usr/bin/env python3
"""
ROS client that queries a remote velocity service over HTTP and forwards the
returned joint velocities to the Dinova base (Dingo) and arm (Kinova).

Run (after `roslaunch dinova_bringup dinova.launch` is up):
    python3 dummy_client.py
    python3 dummy_client.py _server_url:=http://<ip>:4000 _robot_name:=dingo2

Topics (robot_name defaults to $ROBOT_NAME, else "dingo2"):
    sub /<robot_name>/dinova/omni_states_vicon  (or omni_states without vicon)
    pub /<robot_name>/cmd_vel                   geometry_msgs/Twist   base
    pub /<robot_name>/kinova/command            Float64MultiArray     arm, rad/s

Request  (POST <server_url><endpoint>, JSON):
    {
        "time": float,                 # ROS time [s]
        "first_call": bool,
        "joint_names": [str, ...],     # omni_joint_x, omni_joint_y, omni_joint_theta, joint_1..joint_6, ...
        "q":  [float, ...],            # [x, y, theta, q1..q6, ...]  (m, rad)
        "dq": [float, ...]             # corresponding velocities
    }

Response (JSON):
    {
        "base_velocity": [vx, vy, wz],     # m/s, m/s, rad/s
        "arm_velocity":  [dq1, ..., dq6]   # rad/s
    }

The frame of base_velocity is set by the ~base_velocity_frame param:
    "robot" (default): passed to cmd_vel as is (Dingo body frame)
    "world": rotated into the body frame using the current base heading theta

Safety: on any failure (no state yet, request error, malformed response) zero
velocities are published. The kinova driver holds the last command, so this is
required to stop the arm. Velocities are clipped to ~max_base_lin_vel,
~max_base_ang_vel and ~max_arm_vel. The joystick still overrides the base via
twist_mux (cmd_vel has the lowest priority), and the kinova driver's joystick
emergency button still stops the arm.
"""

import math
import os

import numpy as np
import requests
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

N_BASE = 3
N_ARM = 6


class VelocityClient:
    def __init__(self):
        self.server_url = rospy.get_param("~server_url", "http://192.168.0.122:4000").rstrip("/") #http://192.168.5.187
        self.endpoint = rospy.get_param("~endpoint", "/velocity_action")
        self.frequency = rospy.get_param("~frequency", 20.0)
        self.timeout = rospy.get_param("~request_timeout", 1.0 / self.frequency)
        self.base_frame = rospy.get_param("~base_velocity_frame", "robot")
        self.max_lin = rospy.get_param("~max_base_lin_vel", 0.3)
        self.max_ang = rospy.get_param("~max_base_ang_vel", 0.5)
        self.max_arm = rospy.get_param("~max_arm_vel", 0.5)
        self.state_timeout = rospy.get_param("~state_timeout", 0.5)

        if self.base_frame not in ("robot", "world"):
            raise ValueError(f"~base_velocity_frame must be 'robot' or 'world', got '{self.base_frame}'")

        robot = rospy.get_param("~robot_name", os.environ.get("ROBOT_NAME", "dingo2"))
        use_vicon = rospy.get_param(f"/{robot}/vicon/use_vicon", False)
        state_topic = rospy.get_param(
            "~state_topic", f"/{robot}/dinova/omni_states_vicon" if use_vicon else f"/{robot}/dinova/omni_states"
        )

        self.state = None
        self.state_stamp = None
        self.session = requests.Session()

        self.pub_base = rospy.Publisher(f"/{robot}/cmd_vel", Twist, queue_size=1)
        self.pub_arm = rospy.Publisher(f"/{robot}/kinova/command", Float64MultiArray, queue_size=1)
        rospy.Subscriber(state_topic, JointState, self.state_cb, queue_size=1)

        rospy.on_shutdown(self.stop)
        rospy.loginfo(
            f"[velocity_client] server={self.server_url}{self.endpoint} state={rospy.resolve_name(state_topic)} "
            f"rate={self.frequency}Hz base_frame={self.base_frame}"
        )

    def state_cb(self, msg):
        self.state = msg
        self.state_stamp = rospy.Time.now()

    def state_is_fresh(self):
        return (
            self.state is not None
            and (rospy.Time.now() - self.state_stamp).to_sec() < self.state_timeout
        )

    def publish(self, base_vel, arm_vel):
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = base_vel
        self.pub_base.publish(twist)
        self.pub_arm.publish(Float64MultiArray(data=list(arm_vel)))

    def stop(self):
        self.publish(np.zeros(N_BASE), np.zeros(N_ARM))

    def parse_response(self, result):
        base = np.asarray(result["base_velocity"], dtype=float)
        arm = np.asarray(result["arm_velocity"], dtype=float)
        if base.shape != (N_BASE,) or arm.shape != (N_ARM,):
            raise ValueError(f"expected {N_BASE} base / {N_ARM} arm velocities, got {base.shape} / {arm.shape}")
        if not (np.all(np.isfinite(base)) and np.all(np.isfinite(arm))):
            raise ValueError("non-finite velocity in response")

        if self.base_frame == "world":
            theta = self.state.position[2]
            c, s = math.cos(theta), math.sin(theta)
            vx_w, vy_w = base[0], base[1]
            base[0] = c * vx_w + s * vy_w
            base[1] = -s * vx_w + c * vy_w

        lin = np.linalg.norm(base[:2])
        if lin > self.max_lin:
            base[:2] *= self.max_lin / lin
        base[2] = np.clip(base[2], -self.max_ang, self.max_ang)
        arm = np.clip(arm, -self.max_arm, self.max_arm)
        return base, arm

    def run(self):
        rate = rospy.Rate(self.frequency)
        first_call = True
        while not rospy.is_shutdown():
            if not self.state_is_fresh():
                rospy.logwarn_throttle(2.0, "[velocity_client] waiting for robot state, sending zero velocities")
                self.stop()
                rate.sleep()
                continue

            state = self.state
            request = {
                "time": rospy.Time.now().to_sec(),
                "first_call": first_call,
                "joint_names": list(state.name),
                "q": list(state.position),
                "dq": list(state.velocity),
            }
            try:
                response = self.session.post(self.server_url + self.endpoint, json=request, timeout=self.timeout)
                response.raise_for_status()
                base_vel, arm_vel = self.parse_response(response.json())
                print("base_vel:", base_vel)
                print("arm_vel:", arm_vel)
                first_call = False
            except (requests.RequestException, ValueError, KeyError, TypeError) as e:
                rospy.logwarn_throttle(2.0, f"[velocity_client] request failed, sending zero velocities: {e}")
                self.stop()
                rate.sleep()
                continue

            self.publish(base_vel, arm_vel)
            rate.sleep()


def main():
    rospy.init_node("velocity_client")
    VelocityClient().run()


if __name__ == "__main__":
    main()
