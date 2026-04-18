#!/usr/bin/env python3
import rospy
import actionlib
import os
import tf
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, tan
from os import system, name
import time
import re
import fileinput
import sys
import argparse
import random
import matplotlib.animation as animation
from datetime import datetime
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle
from itertools import product
from utils.astar import Astar
from utils.utils import plot_a_car, get_discretized_thetas, round_theta, same_point
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import shutil
import copy
import moveit_commander
from utils.grid import Grid
from utils.car import SimpleCar
from utils.environment import Environment
from utils.dubins_path import DubinsPath
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# Import here the packages used in your codes

""" ----------------------------------------------------------------------------------
Mission planner for Autonomos robots: TTK4192,NTNU. 
Date:20.03.23
characteristics: AI planning,GNC, hybrid A*, ROS.
robot: Turtlebot3
version: 1.1
""" 

# 1. Make it possible to run the plan.py file in this folder
# 2. Read the plan
# 3. Parse the plan
# 4. Send commands to GNC, like State machine 


# 1) Program here your AI planner 
global WAYPOINTS
WAYPOINTS = [[0.3,0.3,0],[1.6,0.3,0],[2.9,  1.3,0], [3.36, 2.7,0], [5.15, 0.25,0], [0.87, 2.56,0], [3.86, 1.8,0]]

WP_MAP = {
    'waypoint0': [0.3,  0.3,  0],   # valid as-is
    'waypoint1': [1.6,  0.45,  np.pi/2],   # valid as-is
    'waypoint2': [3.11,  1.3,  -np.pi/2],   # was [3.41,1.0]: inside box obstacle at x=3.06-3.76, y=0.7-1.1
    'waypoint3': [2.9,  2.8,  np.pi/2],   # was [3.41,1.0]: inside box obstacle at x=3.06-3.76, y=0.7-1.1
    'waypoint4': [4.2,  0.5,  0],   # was [5.15,0.25]: too close to right wall and floor step
    'waypoint5': [0.4, 2.4,  np.pi/2],   # was [0.87,2.56]: car top edge clipped top-wall safe zone
    'waypoint6': [3.46,  1.6,  np.pi/2],
    'safepoint': [2.5, 0.8, 0]   
}

NAVIGATION_STATUS_TEXT = {
    GoalStatus.PENDING: "pending",
    GoalStatus.ACTIVE: "active",
    GoalStatus.PREEMPTED: "preempted",
    GoalStatus.SUCCEEDED: "succeeded",
    GoalStatus.ABORTED: "aborted",
    GoalStatus.REJECTED: "rejected",
    GoalStatus.PREEMPTING: "preempting",
    GoalStatus.RECALLING: "recalling",
    GoalStatus.RECALLED: "recalled",
    GoalStatus.LOST: "lost",
}

MAX_NAVIGATION_RETRIES = 1
MAX_SAFEPOINT_RECOVERIES = 1

def ensure_ros_node(name='mission_planner'):
    if not rospy.core.is_initialized():
        rospy.init_node(name, anonymous=False)

def set_navigation_goal(x, y, yaw=0.0, frame_id="map", timeout=180.0):
    """
    Send a goal to the navigation stack. This uses the same move_base action
    path that RViz's 2D Nav Goal and the SLAM navigation setup use.
    """
    ensure_ros_node()

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    if not client.wait_for_server(rospy.Duration(30.0)):
        rospy.logerr("move_base action server is not available. Start turtlebot3_slam.launch or another launch file that brings up move_base.")
        return GoalStatus.LOST

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = frame_id
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = 0.0

    quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
    goal.target_pose.pose.orientation.x = quaternion[0]
    goal.target_pose.pose.orientation.y = quaternion[1]
    goal.target_pose.pose.orientation.z = quaternion[2]
    goal.target_pose.pose.orientation.w = quaternion[3]

    rospy.loginfo(f"Sending navigation goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}, frame={frame_id}")
    client.send_goal(goal)

    finished = client.wait_for_result(rospy.Duration(timeout))
    if not finished:
        rospy.logwarn(f"Navigation goal timed out after {timeout:.1f}s; cancelling goal.")
        client.cancel_goal()
        return GoalStatus.LOST

    status = client.get_state()
    status_text = NAVIGATION_STATUS_TEXT.get(status, str(status))
    if status == GoalStatus.SUCCEEDED:
        rospy.loginfo("Navigation goal reached.")
        return status

    rospy.logwarn(f"Navigation goal finished with status: {status_text}")
    return status

def get_action_waypoint(action):
    """Return the waypoint an action depends on, when it is encoded in args."""
    action_name = action.get('action')
    action_args = action.get('args', [])

    if action_name == "move_robot" and len(action_args) > 2:
        return action_args[2]

    for arg in action_args[1:]:
        if arg.startswith("waypoint"):
            return arg

    return None

def defer_waypoint_actions(plan, move_index, waypoint):
    """
    Move a failed navigation action and the immediately following actions at
    that destination waypoint to the back of the remaining plan.
    """
    deferred = [plan.pop(move_index)]

    while move_index < len(plan):
        next_action = plan[move_index]
        if next_action.get('action') == "move_robot":
            break
        if get_action_waypoint(next_action) != waypoint:
            break
        deferred.append(plan.pop(move_index))

    for action in deferred:
        action['_navigation_retries'] = action.get('_navigation_retries', 0) + 1

    plan.extend(deferred)
    return deferred

def get_safepoint_waypoint(failed_wp):
    if failed_wp == "safepoint":
        return None
    if "safepoint" not in WP_MAP:
        return None
    return "safepoint"
"""
Graph plan ---------------------------------------------------------------------------
"""
def run_stp_planner(domain_file, problem_file):
    venv_python    = "/home/ttk4192/catkin_ws/src/temporal-planning-main/venv/bin/python2.7"
    planner_script = "/home/ttk4192/catkin_ws/src/temporal-planning-main/temporal-planning/bin/plan.py"
    planner_dir    = "/home/ttk4192/catkin_ws/src/temporal-planning-main/temporal-planning"
    plan_file      = os.path.join(planner_dir, "tmp_sas_plan.1")
 
    # Mimic what 'source activate' does — prepend venv bin to PATH
    venv_bin = "/home/ttk4192/catkin_ws/src/temporal-planning-main/venv/bin"
    env = os.environ.copy()
    env["PATH"] = venv_bin + ":" + env["PATH"]
    env["VIRTUAL_ENV"] = "/home/ttk4192/catkin_ws/src/temporal-planning-main"
 
    cmd = [
        venv_python, planner_script, "stp-2",
        domain_file,
        problem_file
    ]
 
    print("[Planner] Running STP planner...")
 
    result = subprocess.run(
        cmd,
        cwd=planner_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env  # <-- pass the modified environment
    )
 
    print(result.stdout.decode())
    if result.returncode != 0:
        print("[Planner] Error:")
        print(result.stderr.decode())
        return None
 
    if not os.path.exists(plan_file):
        print("[Planner] Plan file not found — planner may have failed.")
        return None
 
    return parse_plan(plan_file)

def parse_plan(plan_file):
    """
    Parses the tmp_sas_plan.1 file and returns a list of actions.
    Each action is a dict with keys: time, action, args, duration
    """
    actions = []
    with open(plan_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith(';'):
                continue
            # Format: <time>: (<action> <args>) [<duration>]
            # Example: 0.000: (move turtlebot0 wp0 wp1 d01) [5.000]
            try:
                time_part, rest = line.split(':', 1)
                action_part = rest.strip().lstrip('(').split(')')[0]
                tokens = action_part.strip().split()
                action_name = tokens[0]
                action_args = tokens[1:]
                duration = None
                if '[' in line:
                    duration = float(line.split('[')[1].rstrip(']'))
                actions.append({
                    'time':     float(time_part.strip()),
                    'action':   action_name,
                    'args':     action_args,
                    'duration': duration
                })
            except Exception as e:
                print(f"[Planner] Could not parse line: {line} ({e})")
 
    print(f"[Planner] Found {len(actions)} actions in plan.")
    for a in actions:
        print(f"  t={a['time']:.3f}: {a['action']} {' '.join(a['args'])}")
    return actions

class GraphPlan(object):
    def __init__(self, domain, problem):
        self.independentActions = []
        self.noGoods = []
        self.graph = []

    def graphPlan(self):
        # initialization
        initState = self.initialState

    def extract(self, Graph, subGoals, level):

        if level == 0:
            return []
        if subGoals in self.noGoods[level]:
            return None
        plan = self.gpSearch(Graph, subGoals, [], level)
        if plan is not None:
            return plan
        self.noGoods[level].append([subGoals])
        return None


#2) GNC module (path-followig and PID controller for the robot)
"""  Robot GNC module ----------------------------------------------------------------------
"""
class PID:
    """
    Discrete PID control
    """
    def __init__(self, P=0.0, I=0.0, D=0.0, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.Derivator = Derivator
        self.Integrator = Integrator
        self.Integrator_max = Integrator_max
        self.Integrator_min = Integrator_min
        self.set_point = 0.0
        self.error = 0.0

    def update(self, current_value):
        PI = 3.1415926535897
        self.error = self.set_point - current_value
        if self.error > pi:  # specific design for circular situation
            self.error = self.error - 2*pi
        elif self.error < -pi:
            self.error = self.error + 2*pi
        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * ( self.error - self.Derivator)
        self.Derivator = self.error
        self.Integrator = self.Integrator + self.error
        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min
        self.I_value = self.Integrator * self.Ki
        PID = self.P_value + self.I_value + self.D_value
        return PID

    def setPoint(self, set_point):
        self.set_point = set_point
        self.Derivator = 0
        self.Integrator = 0

    def setPID(self, set_P=0.0, set_I=0.0, set_D=0.0):
        self.Kp = set_P
        self.Ki = set_I
        self.Kd = set_D

class turtlebot_move():
    """
    Path-following module
    """
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('turtlebot_move', anonymous=False)

        rospy.loginfo("Press CTRL + C to terminate")
        rospy.on_shutdown(self.stop)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.pid_theta = PID(0,0,0)  # initialization

        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback) # subscribing to the odometer
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)        # reading vehicle speed
        self.vel = Twist()
        self.rate = rospy.Rate(10)
        self.counter = 0
        self.trajectory = list()

        # track a sequence of waypoints
        for point in WAYPOINTS:
            self.move_to_point(point[0], point[1])
            rospy.sleep(1)
        self.stop()
        rospy.logwarn("Action done.")

        # plot trajectory
        data = np.array(self.trajectory)
        np.savetxt('trajectory.csv', data, fmt='%f', delimiter=',')
        plt.plot(data[:,0],data[:,1])
        plt.show()


    def move_to_point(self, x, y):
        # Here must be improved the path-following ---
        # Compute orientation for angular vel and direction vector for linear velocity

        diff_x = x - self.x
        diff_y = y - self.y
        direction_vector = np.array([diff_x, diff_y])
        direction_vector = direction_vector/sqrt(diff_x*diff_x + diff_y*diff_y)  # normalization
        theta = atan2(diff_y, diff_x)

        # We should adopt different parameters for different kinds of movement
        self.pid_theta.setPID(1, 0, 0)     # P control while steering
        self.pid_theta.setPoint(theta)
        rospy.logwarn("### PID: set target theta = " + str(theta) + " ###")

        
        # Adjust orientation first
        while not rospy.is_shutdown():
            angular = self.pid_theta.update(self.theta)
            if abs(angular) > 0.2:
                angular = angular/abs(angular)*0.2
            if abs(angular) < 0.01:
                break
            self.vel.linear.x = 0
            self.vel.angular.z = angular
            self.vel_pub.publish(self.vel)
            self.rate.sleep()

        # Have a rest
        self.stop()
        self.pid_theta.setPoint(theta)
        self.pid_theta.setPID(1, 0.02, 0.2)  # PID control while moving

        # Move to the target point
        while not rospy.is_shutdown():
            diff_x = x - self.x
            diff_y = y - self.y
            vector = np.array([diff_x, diff_y])
            linear = np.dot(vector, direction_vector) # projection
            if abs(linear) > 0.2:
                linear = linear/abs(linear)*0.2

            angular = self.pid_theta.update(self.theta)
            if abs(angular) > 0.2:
                angular = angular/abs(angular)*0.2

            if abs(linear) < 0.01 and abs(angular) < 0.01:
                break
            self.vel.linear.x = 1.5*linear   # Here can adjust speed
            self.vel.angular.z = angular
            self.vel_pub.publish(self.vel)
            self.rate.sleep()
        self.stop()
    def stop(self):
        self.vel.linear.x = 0
        self.vel.angular.z = 0
        self.vel_pub.publish(self.vel)
        rospy.sleep(1)

    def odom_callback(self, msg):
        # Get (x, y, theta) specification from odometry topic
        quarternion = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,\
                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quarternion)
        self.theta = yaw
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Make messages saved and prompted in 5Hz rather than 100Hz
        self.counter += 1
        if self.counter == 20:
            self.counter = 0
            self.trajectory.append([self.x,self.y])
            #rospy.loginfo("odom: x=" + str(self.x) + ";  y=" + str(self.y) + ";  theta=" + str(self.theta))



# 3) Program here your path-finding algorithm
""" Hybrid A-star pathfinding --------------------------------------------------------------------
"""
class HybridAstar:
    """ Hybrid A* search procedure. """
    def __init__(self, car, grid, reverse, unit_theta=pi/12, dt=1e-2, check_dubins=1):
        self.car = car
        self.grid = grid
        self.reverse = reverse
        self.unit_theta = unit_theta
        self.dt = dt
        self.check_dubins = check_dubins

        self.start = self.car.start_pos
        self.goal = self.car.end_pos

        self.r = self.car.l / tan(self.car.max_phi)
        self.drive_steps = int(sqrt(2)*self.grid.cell_size/self.dt) + 1
        self.arc = self.drive_steps * self.dt
        self.phil = [-self.car.max_phi, 0, self.car.max_phi]
        self.ml = [1, -1]

        if reverse:
            self.comb = list(product(self.ml, self.phil))
        else:
            self.comb = list(product([1], self.phil))

        self.dubins = DubinsPath(self.car)
        self.astar = Astar(self.grid, self.goal[:2])
        
        self.w1 = 0.95 # weight for astar heuristic
        self.w2 = 0.05 # weight for simple heuristic
        self.w3 = 0.30 # weight for extra cost of steering angle change
        self.w4 = 0.10 # weight for extra cost of turning
        self.w5 = 1.00 # weight for extra cost of reversing

        self.thetas = get_discretized_thetas(self.unit_theta)
    
    def construct_node(self, pos):
        """ Create node for a pos. """

        theta = pos[2]
        pt = pos[:2]

        theta = round_theta(theta % (2*pi), self.thetas)
        
        cell_id = self.grid.to_cell_id(pt)
        grid_pos = cell_id + [theta]

        node = Node(grid_pos, pos)

        return node
    
    def simple_heuristic(self, pos):
        """ Heuristic by Manhattan distance. """

        return abs(self.goal[0]-pos[0]) + abs(self.goal[1]-pos[1])
        
    def astar_heuristic(self, pos):
        """ Heuristic by standard astar. """

        result = self.astar.search_path(pos[:2])
        if result is None:
            return self.simple_heuristic(pos[:2])
        h1 = result * self.grid.cell_size
        h2 = self.simple_heuristic(pos[:2])

        return self.w1*h1 + self.w2*h2

    def get_children(self, node, heu, extra):
        """ Get successors from a state. """

        children = []
        for m, phi in self.comb:

            # don't immediately reverse on the same arc (would retrace exact path)
            if node.m and node.phi == phi and node.m*m == -1:
                continue
            if node.m and node.m == 1 and m == -1:
                continue
            pos = node.pos
            branch = [m, pos[:2]]

            for _ in range(self.drive_steps):
                pos = self.car.step(pos, phi, m)
                branch.append(pos[:2])

            # check safety of route-----------------------
            pos1 = node.pos if m == 1 else pos
            pos2 = pos if m == 1 else node.pos
            if phi == 0:
                safe = self.dubins.is_straight_route_safe(pos1, pos2)
            else:
                d, c, r = self.car.get_params(pos1, phi)
                safe = self.dubins.is_turning_route_safe(pos1, pos2, d, c, r)
            # --------------------------------------------
            
            if not safe:
                continue
            
            child = self.construct_node(pos)
            child.phi = phi
            child.m = m
            child.parent = node
            child.g = node.g + self.arc
            child.g_ = node.g_ + self.arc

            if extra:
                # extra cost for changing steering angle
                if phi != node.phi:
                    child.g += self.w3 * self.arc
                
                # extra cost for turning
                if phi != 0:
                    child.g += self.w4 * self.arc
                
                # extra cost for reverse
                if m == -1:
                    child.g += self.w5 * self.arc

                # extra cost for direction change (forward <-> reverse)
                if node.m and node.m != m:
                    child.g += self.w5 * self.arc

            if heu == 0:
                child.f = child.g + self.simple_heuristic(child.pos)
            if heu == 1:
                child.f = child.g + self.astar_heuristic(child.pos)
            
            children.append([child, branch])

        return children
    
    def best_final_shot(self, open_, closed_, best, cost, d_route, n=10):
        """ Search best final shot in open set. """

        open_.sort(key=lambda x: x.f, reverse=False)

        for t in range(min(n, len(open_))):
            best_ = open_[t]
            solutions_ = self.dubins.find_tangents(best_.pos, self.goal)
            d_route_, cost_, valid_ = self.dubins.best_tangent(solutions_)
        
            if valid_ and cost_ + best_.g_ < cost + best.g_:
                best = best_
                cost = cost_
                d_route = d_route_
        
        if best in open_:
            open_.remove(best)
            closed_.append(best)
        
        return best, cost, d_route
    
    def backtracking(self, node):
        """ Backtracking the path. """

        route = []
        while node.parent:
            route.append((node.pos, node.phi, node.m))
            node = node.parent
        
        return list(reversed(route))
    
    def search_path(self, heu=1, extra=False):
        """ Hybrid A* pathfinding. """

        root = self.construct_node(self.start)
        root.g = float(0)
        root.g_ = float(0)
        
        if heu == 0:
            root.f = root.g + self.simple_heuristic(root.pos)
        if heu == 1:
            root.f = root.g + self.astar_heuristic(root.pos)

        closed_ = []
        open_ = [root]

        goal_node = self.construct_node(self.goal)

        count = 0
        while open_:
            count += 1
            best = min(open_, key=lambda x: x.f)

            open_.remove(best)
            closed_.append(best)

            # check if A* expansion reached the goal cell directly
            if best.grid_pos == goal_node.grid_pos:
                route = self.backtracking(best)
                path = self.car.get_path(self.start, route)
                cost = best.g_
                print('Path found (direct): {}'.format(round(cost, 2)))
                print('Total iteration:', count)
                return path, closed_

            # check dubins path
            if count % self.check_dubins == 0:
                solutions = self.dubins.find_tangents(best.pos, self.goal)
                d_route, cost, valid = self.dubins.best_tangent(solutions)

                if valid:
                    best, cost, d_route = self.best_final_shot(open_, closed_, best, cost, d_route)
                    route = self.backtracking(best) + d_route
                    path = self.car.get_path(self.start, route)
                    cost += best.g_
                    print('Shortest path (Dubins): {}'.format(round(cost, 2)))
                    print('Total iteration:', count)

                    return path, closed_

            children = self.get_children(best, heu, extra)

            for child, branch in children:

                if child in closed_:
                    continue

                if child not in open_:
                    best.branches.append(branch)
                    open_.append(child)

                elif child.g < open_[open_.index(child)].g:
                    best.branches.append(branch)

                    c = open_[open_.index(child)]
                    p = c.parent
                    for b in p.branches:
                        if same_point(b[-1], c.pos[:2]):
                            p.branches.remove(b)
                            break
                    
                    open_.remove(child)
                    open_.append(child)

        return None, closed_


def plot_search_space(env, car, grid, closed_, grid_on):
    """ Plot explored search space when no valid path was found. """

    branches = []
    bcolors = []
    for node in closed_:
        for b in node.branches:
            branches.append(b[1:])
            bcolors.append('y' if b[0] == 1 else 'b')

    all_x = [ob.x + ob.w for ob in env.obs] + [0]
    all_y = [ob.y + ob.h for ob in env.obs] + [0]
    room_w = max(all_x) + 0.3
    room_h = max(all_y) + 0.3

    fig, ax = plt.subplots(figsize=(max(6, 6*room_w/room_h), 6))
    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_h)
    ax.set_aspect("equal")
    ax.set_title("No valid path found — explored search space", color='red')

    if grid_on:
        ax.set_xticks(np.arange(0, room_w, grid.cell_size))
        ax.set_yticks(np.arange(0, room_h, grid.cell_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        plt.grid(which='both')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))

    start_state = car.get_car_state(car.start_pos)
    end_state   = car.get_car_state(car.end_pos)
    ax = plot_a_car(ax, end_state.model)
    ax = plot_a_car(ax, start_state.model)
    ax.plot(car.start_pos[0], car.start_pos[1], 'ro', markersize=6)
    ax.plot(car.end_pos[0],   car.end_pos[1],   'rx', markersize=8, markeredgewidth=2)

    if branches:
        lc = LineCollection(branches, linewidth=0.6, colors=bcolors, alpha=0.7)
        ax.add_collection(lc)

    # mark every explored node position
    xs = [n.pos[0] for n in closed_]
    ys = [n.pos[1] for n in closed_]
    ax.scatter(xs, ys, s=4, c='cyan', zorder=3, label=f'explored ({len(closed_)} nodes)')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def main_hybrid_a(heu,start_pos, end_pos,reverse, extra, grid_on):

    tc = map_grid()
    env = Environment(tc.obs)
    car = SimpleCar(env, start_pos, end_pos, l=0.19)
    grid = Grid(env)

    hastar = HybridAstar(car, grid, reverse)

    t = time.time()
    path, closed_ = hastar.search_path(heu, extra)
    print('Total time: {}s'.format(round(time.time()-t, 3)))

    if not path:
        print('No valid path!')
        if closed_:
            print('Plotting explored search space ({} nodes)...'.format(len(closed_)))
            plot_search_space(env, car, grid, closed_, grid_on)
        return
    # a post-processing is required to have path list
    path = path[::5] + [path[-1]]
    branches = []
    bcolors = []
    for node in closed_:
        for b in node.branches:
            branches.append(b[1:])
            bcolors.append('y' if b[0] == 1 else 'b')

    xl, yl = [], []
    xl_np1,yl_np1=[],[]
    carl = []
    dt_s=int(25)  # samples for gazebo simulator
    for i in range(len(path)):
        xl.append(path[i].pos[0])
        yl.append(path[i].pos[1])
        carl.append(path[i].model[0])
        if i==0 or i==len(path):
            xl_np1.append(path[i].pos[0])
            yl_np1.append(path[i].pos[1])            
        elif dt_s*i<len(path):
            xl_np1.append(path[i*dt_s].pos[0])
            yl_np1.append(path[i*dt_s].pos[1])      
    # defining way-points
    xl_np=np.array(xl_np1)
    xl_np=xl_np
    yl_np=np.array(yl_np1)
    yl_np=yl_np
    global WAYPOINTS
    WAYPOINTS=np.column_stack([xl_np,yl_np])
    #print(WAYPOINTS)
    
    start_state = car.get_car_state(car.start_pos)
    end_state = car.get_car_state(car.end_pos)

    # plot and annimation
    # Compute actual room bounds from obstacles with a small margin
    all_x = [ob.x + ob.w for ob in env.obs] + [0]
    all_y = [ob.y + ob.h for ob in env.obs] + [0]
    room_w = max(all_x) + 0.3
    room_h = max(all_y) + 0.3
    fig, ax = plt.subplots(figsize=(max(6, 6*room_w/room_h), 6))
    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_h)
    ax.set_aspect("equal")

    if grid_on:
        ax.set_xticks(np.arange(0, room_w, grid.cell_size))
        ax.set_yticks(np.arange(0, room_h, grid.cell_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        plt.grid(which='both')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
    
    ax.plot(car.start_pos[0], car.start_pos[1], 'ro', markersize=6)
    ax = plot_a_car(ax, end_state.model)
    ax = plot_a_car(ax, start_state.model)

    _branches = LineCollection([], linewidth=1)
    ax.add_collection(_branches)

    _path, = ax.plot([], [], color='lime', linewidth=2)
    _carl = PatchCollection([])
    ax.add_collection(_carl)
    _path1, = ax.plot([], [], color='w', linewidth=2)
    _car = PatchCollection([])
    ax.add_collection(_car)
    
    frames = len(branches) + len(path) + 1

    def init():
        _branches.set_paths([])
        _path.set_data([], [])
        _carl.set_paths([])
        _path1.set_data([], [])
        _car.set_paths([])

        return _branches, _path, _carl, _path1, _car

    def animate(i):

        edgecolor = ['k']*5 + ['r']
        facecolor = ['y'] + ['k']*4 + ['r']

        if i < len(branches):
            _branches.set_paths(branches[:i+1])
            _branches.set_color(bcolors)
        
        else:
            _branches.set_paths(branches)

            j = i - len(branches)

            _path.set_data(xl[min(j, len(path)-1):], yl[min(j, len(path)-1):])

            sub_carl = carl[:min(j+1, len(path))]
            _carl.set_paths(sub_carl[::4])
            _carl.set_edgecolor('k')
            _carl.set_facecolor('m')
            _carl.set_alpha(0.1)
            _carl.set_zorder(3)

            _path1.set_data(xl[:min(j+1, len(path))], yl[:min(j+1, len(path))])
            _path1.set_zorder(3)

            _car.set_paths(path[min(j, len(path)-1)].model)
            _car.set_edgecolor(edgecolor)
            _car.set_facecolor(facecolor)
            _car.set_zorder(3)

        return _branches, _path, _carl, _path1, _car

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                  interval=1, repeat=True, blit=True)

    plt.show()

class Node:
    """ Hybrid A* tree node. """

    def __init__(self, grid_pos, pos):

        self.grid_pos = grid_pos
        self.pos = pos
        self.g = None
        self.g_ = None
        self.f = None
        self.parent = None
        self.phi = 0
        self.m = None
        self.branches = []

    def __eq__(self, other):

        return self.grid_pos == other.grid_pos
    
    def __hash__(self):

        return hash((self.grid_pos))
    
class map_grid:
    """ Here the obstacles are defined for a 20x20 map. """
    def __init__(self):

        self.start_pos2 = [4, 4, 0]  # default values
        self.end_pos2 = [4, 8, -pi]  # default
        self.obs = [
            #x,y,x-bredde,y-høyde
        
            [0.0, 0.0,   3.3,  0.05],
            [3.275, 0.0, 0.05, 0.2],
            [3.3, 0.175, 1.91, 0.05],
            [0.0, 2.725, 5.21, 0.05],
            [0.0, 0.0,   0.05, 1.0],
            [0.0, 0.975, 0.5,  0.05],
            [0.475, 1.0, 0.05, 0.2],
            [0.0, 1.175, 0.5,  0.05],
            [0.0, 1.2,   0.05, 1.55],
            [5.185, 0.2, 0.05, 2.55],
            [1.2, 1.65, 0.2, 0.4],
            [2.3, 1.65, 0.4, 0.4],
            [3.66, 1.8, 0.4, 0.2],
            [1.35, 0.6, 0.5, 0.2],
            [3.16, 0.8, 0.5, 0.2],
        ]


#4) Program here the turtlebot actions (based in your AI planner)
"""
Turtlebot 3 actions-------------------------------------------------------------------------
"""

class TakePhoto:
    def __init__(self):

        self.bridge = CvBridge()
        self.image_received = False

        # Connect image topic
        img_topic = "/camera/image"
        self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

        # Allow up to one second to connection
        rospy.sleep(1)

    def callback(self, data):

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image_received = True
        self.image = cv_image

    def take_picture(self, img_title):
        if self.image_received:
            # Save an image
            cv2.imwrite(img_title, self.image)
            return True
        else:
            return False
        
def taking_photo_exe():
    # Initialize
    camera = TakePhoto()

    # Default value is 'photo.jpg'
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    img_title = rospy.get_param('~image_title', 'photo'+dt_string+'.jpg')

    if camera.take_picture(img_title):
        rospy.loginfo("Saved image " + img_title)
    else:
        rospy.loginfo("No images received")
	#eog photo.jpg
    # Sleep to give the last log messages time to be sent

	# saving photo in a desired directory
    file_source = '/home/ttk4192/catkin_ws/'
    file_destination = '/home/ttk4192/catkin_ws/src/assigment4_ttk4192/scripts'
    g='photo'+dt_string+'.jpg'

    shutil.move(file_source + g, file_destination)
    rospy.sleep(1)

class TB3Manipulator:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = moveit_commander.MoveGroupCommander("arm")
        self.gripper = moveit_commander.MoveGroupCommander("gripper")

        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)
        self.gripper.set_max_velocity_scaling_factor(1.0)
        self.gripper.set_max_acceleration_scaling_factor(1.0)

    def arm_home(self):
        self.arm.set_named_target("home")
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return ok

    def set_gripper(self, value):
        rospy.loginfo(f"Setting gripper joint to {value}")

        # only target the ACTIVE joint
        self.gripper.set_joint_value_target({"gripper": value})

        ok = self.gripper.go(wait=True)
        self.gripper.stop()
        self.gripper.clear_pose_targets()

        rospy.loginfo(f"Gripper result: {ok}")
        rospy.loginfo(f"Current gripper values: {self.gripper.get_current_joint_values()}")
        return ok

    # def open_gripper(self):
    #     return self.set_gripper(0.01)

    # def close_gripper(self):
    #     return self.set_gripper(-0.01)

    def open_gripper(self):
        self.gripper.set_joint_value_target({"gripper": 0.010})
        self.gripper.go(wait=True)
        self.gripper.stop()

    def close_gripper(self):
        self.gripper.set_joint_value_target({"gripper": -0.010})
        self.gripper.go(wait=True)
        self.gripper.stop()

def use_gripper_exe():
    manip = TB3Manipulator()

    # rospy.loginfo("Moving arm to home")
    # manip.arm_home()
    # rospy.sleep(1)

    rospy.loginfo("Opening gripper")
    manip.open_gripper()
    rospy.sleep(1)

    rospy.loginfo("Closing gripper")
    manip.close_gripper()
    rospy.sleep(1)

#Think this is good, needs testing
def move_robot_from_to(start_pos, end_pos):
    # Execute the move_robot action through SLAM navigation.
    a=0
    while a<3:
        print("Excuting Mr12")
        time.sleep(1)
        a=a+1

    ensure_ros_node()

    print("Executing SLAM navigation goal with move_base")
    frame_id = rospy.get_param("~navigation_frame", "map")
    timeout = rospy.get_param("~navigation_goal_timeout", 180.0)
    status = set_navigation_goal(end_pos[0], end_pos[1], end_pos[2], frame_id, timeout)
    if status != GoalStatus.SUCCEEDED:
        status_text = NAVIGATION_STATUS_TEXT.get(status, str(status))
        rospy.logwarn(f"Failed to reach navigation goal {end_pos}: {status_text}")
    return status
    


#Not done
def Manipulate_OpenManipulator_x():
    print("Executing manipulate a weight")
    time.sleep(5)

def making_turn_exe():
    print("Executing Make a turn")
    time.sleep(1)
    #Starts a new node
    #rospy.init_node('turtlebot_move', anonymous=True)
    velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    # Receiveing the user's input
    print("Let's rotate your robot")
    #speed = input("Input your speed (degrees/sec):")
    #angle = input("Type your distance (degrees):")
    #clockwise = input("Clockwise?: ") #True or false

    speed = 5
    angle = 180
    clockwise = True

    #Converting from angles to radians
    angular_speed = speed*2*pi/360
    relative_angle = angle*2*pi/360

    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW
    if clockwise:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_angle = 0   #should be from the odometer

    while(current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)

    #Forcing our robot to stop
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    #rospy.spin()

#There are no Ir-topic as far as I can see, so it should be the same as take photo
def check_pump_picture_ir_waypoint0():
    a=0
    while a<3:
        print("Taking IR picture at waypoint0 ...")
        time.sleep(1)
        a=a+1
    taking_photo_exe()
    time.sleep(5)

#Think this is good.
def check_seals_valve_picture_eo(WP):
    a=0
    while a<3:
        print(f"Taking EO picture at {WP} ...")
        time.sleep(1)
        a=a+1
    taking_photo_exe()
    time.sleep(5)

# Charging battery, dont know if we need to subscribe to topics here.
def charge_battery_waypoint0():
    print("chargin battery")
    time.sleep(5)






# Define the global varible: WAYPOINTS  Wpts=[[x_i,y_i]];



# 5) Program here the main commands of your mission planner code
""" Main code ---------------------------------------------------------------------------
"""
if __name__ == '__main__':
    try:
        print()
        print("************ TTK4192 - Assigment 4 **************************")
        print()
        print("AI planners: GraphPlan")
        print("Navigation: SLAM map + move_base goals")
        print("GNC Controller: move_base")
        print("Robot: Turtlebot3 waffle-pi")
        print("date: 20.03.23")
        print()
        print("**************************************************************")
        print()
 
        # WAYPOINTS = [[1.99,2.45],[2.99,2.45],]
        # turtlebot_move()
        # check_seals_valve_picture_eo(1)
        # use_gripper_exe()

        plan = run_stp_planner("/home/ttk4192/catkin_ws/src/temporal-planning-main/temporal-planning/domains/ttk4192/domain/PDDL_domain.pddl",
                         "/home/ttk4192/catkin_ws/src/temporal-planning-main/temporal-planning/domains/ttk4192/problem/PDDL_problem.pddl")
 
        if plan:
            print("\n[Planner] Plan summary:")
            for step, a in enumerate(plan, 1):
                print(f"  {step}: {a['action']} {' '.join(a['args'])}")

        # 5.3) Start mission execution
        print("")
        print("Starting mission execution")
        
        # Start simulations with battery = 100%
        battery = 100
        task_finished = 0
        task_total = len(plan)
        max_navigation_retries = rospy.get_param("~max_navigation_retries", MAX_NAVIGATION_RETRIES)
        max_safepoint_recoveries = rospy.get_param("~max_safepoint_recoveries", MAX_SAFEPOINT_RECOVERIES)
        i_ini = 0

        while i_ini < len(plan):
            action = plan[i_ini]
            action_name = action['action']
            action_args = action['args']

            print(f"\n[Mission] Step {i_ini+1}/{len(plan)}: {action_name} {' '.join(action_args)}")

            if action_name == "move_robot":
                # args: [robot, from_wp, to_wp, route]
                from_wp = action_args[1]
                to_wp   = action_args[2]
                print(f"[Mission] Moving robot from {from_wp} to {to_wp}")
                navigation_status = move_robot_from_to(WP_MAP[from_wp],WP_MAP[to_wp])
                if navigation_status == GoalStatus.ABORTED:
                    retry_count = action.get('_navigation_retries', 0)
                    if retry_count < max_navigation_retries:
                        deferred = defer_waypoint_actions(plan, i_ini, to_wp)
                        deferred_names = [
                            f"{a['action']} {' '.join(a['args'])}" for a in deferred
                        ]
                        print(f"[Mission] Navigation to {to_wp} aborted. Deferring dependent waypoint block to the back of the plan:")
                        for deferred_name in deferred_names:
                            print(f"  - {deferred_name}")
                        print(f"[Mission] Will retry {to_wp} after the remaining steps.")
                        continue

                    safepoint_count = action.get('_safepoint_recoveries', 0)
                    if safepoint_count < max_safepoint_recoveries:
                        safepoint_wp = get_safepoint_waypoint(to_wp)
                        if safepoint_wp is None:
                            raise rospy.ROSException(
                                f"Navigation to {to_wp} aborted and no safepoint recovery waypoint is available."
                            )

                        print(f"[Mission] Navigation to {to_wp} aborted after retry. Moving to {safepoint_wp}, then retrying {to_wp}.")
                        safepoint_status = move_robot_from_to(WP_MAP[from_wp], WP_MAP[safepoint_wp])
                        if safepoint_status != GoalStatus.SUCCEEDED:
                            status_text = NAVIGATION_STATUS_TEXT.get(safepoint_status, str(safepoint_status))
                            raise rospy.ROSException(
                                f"Safepoint recovery waypoint {safepoint_wp} failed with status: {status_text}"
                            )

                        action['_safepoint_recoveries'] = safepoint_count + 1
                        action_args[1] = safepoint_wp
                        continue

                    raise rospy.ROSException(
                        f"Navigation to {to_wp} aborted after {retry_count} retries and {action.get('_safepoint_recoveries', 0)} safepoint recoveries."
                    )

                if navigation_status != GoalStatus.SUCCEEDED:
                    status_text = NAVIGATION_STATUS_TEXT.get(navigation_status, str(navigation_status))
                    raise rospy.ROSException(
                        f"Navigation to {to_wp} failed with status: {status_text}"
                    )
                # WAYPOINTS=[WP_MAP[from_wp],WP_MAP[to_wp]]
                # turtlebot_move()
                time.sleep(1)

            elif action_name == "take_picture_pump_ir":
                # args: [robot, waypoint, camera,  pump]
                waypoint = action_args[1] if len(action_args) > 1 else "unknown"
                target   = action_args[3] if len(action_args) > 2 else "unknown"
                print(f"[Mission] Inspecting pump at {waypoint} — target: {target}")
                taking_photo_exe()  # uncomment when ready
                time.sleep(1)

            elif action_name == "check_seals_valve_picture_eo":
                # args: [robot, waypoint, camera, valve]
                waypoint = action_args[1] if len(action_args) > 1 else "unknown"
                target   = action_args[2] if len(action_args) > 2 else "unknown"
                print(f"[Mission] Checking valve EO at {waypoint} — target: {target}")
                taking_photo_exe()  # uncomment when ready
                time.sleep(1)

            elif action_name == "grasp_object":
                # args: [robot, waypoint, arm, object]
                waypoint = action_args[1] if len(action_args) > 1 else "unknown"
                target   = action_args[3] if len(action_args) > 3 else "unknown"
                print(f"[Mission] Grasping {target} at {waypoint}")
                use_gripper_exe()


            elif action_name == "charge_battery":
                # args: [robot, charger_wp, charger]
                print(f"[Mission] Charging battery...")
                battery = 100
                time.sleep(1)

            else:
                print(f"[Mission] Unknown action: {action_name} — skipping")

            task_finished += 1
            # battery -= 5  # rough battery drain per action
            print(f"[Mission] Battery: {battery}%")

            i_ini += 1  # next task

        print("")
        print("--------------------------------------")
        print(f"All {task_finished} tasks were performed successfully")
        time.sleep(10)




        
    except rospy.ROSInterruptException:
        rospy.loginfo("Action terminated.")
