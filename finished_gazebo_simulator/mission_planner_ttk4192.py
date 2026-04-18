#!/usr/bin/env python3
import rospy
import os
import tf
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids Tk/main-thread crash in ROS
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from math import pi, sqrt, atan2, tan, cos, sin
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
from utils.spline_path import SplinePath
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
WAYPOINTS = []   # populated per-leg by main_hybrid_a

# Gazebo model name — must match the name shown in `rostopic echo /gazebo/model_states`
GAZEBO_MODEL_NAME = 'robot'

# Waypoint positions and headings — must match map_grid obstacles and Gazebo world
WP_MAP = {
    'waypoint0': [0.4,  0.3,  0],
    'waypoint1': [1.6,  0.3,  pi/2],
    'waypoint2': [3.41, 1.3,  -pi/2],
    'waypoint3': [3.36, 2.45, pi/2],
    'waypoint4': [4.7,  0.5,  0],
    'waypoint5': [0.87, 2.4,  -pi/2],
    'waypoint6': [3.86, 1.4,  pi/2],
}

MISSION = ['waypoint0', 'waypoint1', 'waypoint2', 'waypoint5', 'waypoint6', 'waypoint3']
"""
Graph plan ---------------------------------------------------------------------------
"""
def run_stp_planner(domain_file, problem_file):
    venv_python    = "/home/appuser/catkin_ws/src/temporal-planning-main/bin/python2.7"
    planner_script = "/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning/bin/plan.py"
    planner_dir    = "/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning"
    plan_file      = os.path.join(planner_dir, "tmp_sas_plan.1")
 
    # Mimic what 'source activate' does — prepend venv bin to PATH
    venv_bin = "/home/appuser/catkin_ws/src/temporal-planning-main/bin"
    env = os.environ.copy()
    env["PATH"] = venv_bin + ":" + env["PATH"]
    env["VIRTUAL_ENV"] = "/home/appuser/catkin_ws/src/temporal-planning-main"
 
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
    Path-following module — continuous pure-pursuit style controller.

    Position source: /gazebo/model_states (ground truth, no drift).
    Falls back to /odom if Gazebo pose is not yet received.

    WAYPOINTS is a list of (x, y, theta, direction) tuples where direction is
    +1.0 (forward) or -1.0 (reverse), inferred from the planned path heading.
    """

    LINEAR_SPEED  = 0.15   # m/s  — forward/reverse driving speed
    K_ANGULAR     = 2.0    # proportional heading gain while driving
    K_ALIGN       = 2.0    # proportional heading gain during final alignment
    MAX_ANGULAR   = 0.5    # rad/s clamp
    ARRIVE_DIST   = 0.05   # m    — switch to next waypoint when closer than this
    ALIGN_TOL     = 0.03   # rad  — final heading tolerance

    def __init__(self, final_heading=None):
        """
        final_heading: desired yaw (rad) at the goal, or None to skip alignment.
        Pass end_pos[2] from WP_MAP so the robot ends up pointing the right way.
        """
        if not rospy.core.is_initialized():
            rospy.init_node('turtlebot_move', anonymous=True)
        rospy.loginfo("Press CTRL + C to terminate")
        rospy.on_shutdown(self.stop)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._gazebo_ok = False   # True once we receive a Gazebo pose

        # Ground-truth pose from Gazebo (preferred — no drift)
        self.gazebo_sub = rospy.Subscriber('/gazebo/model_states', ModelStates,
                                           self.gazebo_callback)
        # Odometry fallback
        self.odom_sub   = rospy.Subscriber('odom', Odometry, self.odom_callback)

        self.vel_pub  = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.vel  = Twist()
        self.rate = rospy.Rate(10)
        self.counter = 0
        self.trajectory = list()

        # Wait for first pose message
        rospy.sleep(0.5)

        for point in WAYPOINTS:
            x, y, _theta_ref, direction = point
            self.move_to_point(x, y, direction)

        # Final heading alignment at the goal
        if final_heading is not None:
            self.align_heading(final_heading)

        self.stop()
        rospy.logwarn("Action done.")

        # Save the driven trajectory (plt.show() is not safe inside ROS callbacks)
        data = np.array(self.trajectory)
        if len(data) > 0:
            np.savetxt('trajectory.csv', data, fmt='%f', delimiter=',')
            fig, ax = plt.subplots()
            ax.plot(data[:, 0], data[:, 1])
            ax.set_title('Driven trajectory')
            fig.savefig('trajectory.png')
            plt.close(fig)
            rospy.loginfo("Trajectory saved to trajectory.csv / trajectory.png")

    def move_to_point(self, x, y, direction):
        """
        Drive toward (x, y) with improved stability.
        direction: +1.0 → forward, -1.0 → reverse.
        """
        TURN_IN_PLACE_THRESH = pi / 3  # 60° — if error is larger, rotate first
        
        while not rospy.is_shutdown():
            dx = x - self.x
            dy = y - self.y
            dist = sqrt(dx * dx + dy * dy)

            if dist < self.ARRIVE_DIST:
                # Stop briefly when arriving
                self.vel.linear.x = 0.0
                self.vel.angular.z = 0.0
                self.vel_pub.publish(self.vel)
                rospy.sleep(0.1)
                break

            # Bearing to target
            bearing = atan2(dy, dx)
            if direction < 0:
                bearing = atan2(-dy, -dx)

            # Heading error normalised to (-pi, pi)
            err = atan2(sin(bearing - self.theta), cos(bearing - self.theta))

            if abs(err) > TURN_IN_PLACE_THRESH:
                # Large heading error — rotate in place first
                angular = self.K_ANGULAR * err
                angular = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, angular))
                self.vel.linear.x = 0.0
                self.vel.angular.z = angular
            else:
                # Normal driving
                angular = self.K_ANGULAR * err
                angular = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, angular))
                
                # Scale speed down when heading error is significant
                speed_scale = max(0.3, 1.0 - abs(err) / TURN_IN_PLACE_THRESH)
                self.vel.linear.x = direction * self.LINEAR_SPEED * speed_scale
                self.vel.angular.z = angular

            self.vel_pub.publish(self.vel)
            self.rate.sleep()

    def align_heading(self, theta_goal):
        """Pure rotation to reach theta_goal within ALIGN_TOL. Called once at the end."""
        rospy.loginfo("Aligning to heading {:.3f} rad".format(theta_goal))
        while not rospy.is_shutdown():
            err = atan2(sin(theta_goal - self.theta), cos(theta_goal - self.theta))
            if abs(err) < self.ALIGN_TOL:
                break
            angular = self.K_ALIGN * err
            angular = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, angular))
            self.vel.linear.x  = 0.0
            self.vel.angular.z = angular
            self.vel_pub.publish(self.vel)
            self.rate.sleep()

    def stop(self):
        self.vel.linear.x  = 0.0
        self.vel.angular.z = 0.0
        self.vel_pub.publish(self.vel)
        rospy.sleep(0.3)

    def gazebo_callback(self, msg):
        """Ground-truth pose from Gazebo — no drift."""
        try:
            idx = msg.name.index(GAZEBO_MODEL_NAME)
        except ValueError:
            return  # model not in list yet
        pose = msg.pose[idx]
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
        self.x = pose.position.x
        self.y = pose.position.y
        self.theta = yaw
        self._gazebo_ok = True

        self.counter += 1
        if self.counter == 20:
            self.counter = 0
            self.trajectory.append([self.x, self.y])

    def odom_callback(self, msg):
        """Odometry fallback — only used if Gazebo pose hasn't arrived yet."""
        if self._gazebo_ok:
            return
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
        self.theta = yaw
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        self.counter += 1
        if self.counter == 20:
            self.counter = 0
            self.trajectory.append([self.x, self.y])



# 3) Program here your path-finding algorithm
""" Hybrid A-star pathfinding --------------------------------------------------------------------
"""
class HybridAstar:
    """ Hybrid A* search procedure. """
    def __init__(self, car, grid, reverse, unit_theta=pi/12, dt=1e-2, check_dubins=1,
                 epsilon=3.0, max_iter=12000, use_spline=True):
        self.car = car
        self.grid = grid
        self.reverse = reverse
        self.unit_theta = unit_theta
        self.dt = dt
        self.check_dubins = check_dubins
        self.epsilon = epsilon   # heuristic inflation: >1 trades optimality for speed
        self.max_iter = max_iter # hard cap on expansions to avoid endless search

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

        if use_spline:
            self.dubins = SplinePath(self.car)
            self.connector_name = 'Spline'
        else:
            self.dubins = DubinsPath(self.car)
            self.connector_name = 'Dubins'
        self.astar = Astar(self.grid, self.goal[:2])
        self.astar.precompute()  # one-time backward Dijkstra from goal

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
        """ Heuristic by precomputed cost-to-go (O(1) lookup). """
        result = self.astar.lookup(pos[:2])
        if result is None:
            return self.epsilon * self.simple_heuristic(pos[:2])
        h1 = result * self.grid.cell_size
        h2 = self.simple_heuristic(pos[:2])
        return self.epsilon * (self.w1*h1 + self.w2*h2)

    def get_children(self, node, heu, extra):
        """ Get successors from a state. """
        children = []
        for m, phi in self.comb:

            # don't immediately reverse on the exact same arc (would retrace)
            if node.m and node.phi == phi and node.m * m == -1:
                continue

            pos = node.pos
            branch = [m, pos[:2]]

            for _ in range(self.drive_steps):
                pos = self.car.step(pos, phi, m)
                branch.append(pos[:2])

            # check safety of route
            if phi == 0:
                pos1 = node.pos if m == 1 else pos
                pos2 = pos if m == 1 else node.pos
                safe = self.dubins.is_straight_route_safe(pos1, pos2)
            else:
                if m == 1:
                    d, c, r = self.car.get_params(node.pos, phi)
                    safe = self.dubins.is_turning_route_safe(node.pos, pos, d, c, r)
                else:
                    # reverse: same geometric centre but effective turn direction flipped
                    _, c, r = self.car.get_params(node.pos, phi)
                    d = -1 if phi > 0 else 1
                    safe = self.dubins.is_turning_route_safe(node.pos, pos, d, c, r)

            if not safe:
                continue

            child = self.construct_node(pos)
            child.phi = phi
            child.m = m
            child.parent = node
            child.g = node.g + self.arc
            child.g_ = node.g_ + self.arc

            # direction-change penalty
            if node.m and node.m != m:
                child.g += self.w5 * self.arc

            if extra:
                if phi != node.phi:
                    child.g += self.w3 * self.arc
                if phi != 0:
                    child.g += self.w4 * self.arc
                if m == -1:
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
            if count > self.max_iter:
                print('Max iterations ({}) reached.'.format(self.max_iter))
                break
            best = min(open_, key=lambda x: x.f)
            open_.remove(best)
            closed_.append(best)

            # direct goal cell match
            if best.grid_pos == goal_node.grid_pos:
                route = self.backtracking(best)
                path = self.car.get_path(self.start, route)
                cost = best.g_
                print('Path found (direct): {}'.format(round(cost, 2)))
                print('Total iteration:', count)
                return path, closed_

            # connector (spline / dubins) final shot
            if count % self.check_dubins == 0:
                solutions = self.dubins.find_tangents(best.pos, self.goal)
                d_route, cost, valid = self.dubins.best_tangent(solutions)
                if valid:
                    best, cost, d_route = self.best_final_shot(open_, closed_, best, cost, d_route)
                    route = self.backtracking(best) + d_route
                    path = self.car.get_path(self.start, route)
                    cost += best.g_
                    print('Shortest path ({}): {}'.format(self.connector_name, round(cost, 2)))
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


def main_hybrid_a(heu, start_pos, end_pos, reverse, extra, grid_on, use_spline=True, safe_dis=0.08):

    tc = map_grid()
    env = Environment(tc.obs, safe_dis=safe_dis)
    car = SimpleCar(env, start_pos, end_pos, l=0.19)
    grid = Grid(env)

    hastar = HybridAstar(car, grid, reverse, use_spline=use_spline)
    print("done with hybrid astar")
    t = time.time()
    path, closed_ = hastar.search_path(heu, extra)
    print('Total time: {}s'.format(round(time.time()-t, 3)))

    if not path:
        print('No valid path!')
        if closed_:
            print('Explored {} nodes before giving up.'.format(len(closed_)))
        return
    # a post-processing is required to have path list
    path = path[::5] + [path[-1]]
    branches = []
    bcolors = []
    for node in closed_:
        for b in node.branches:
            branches.append(b[1:])
            bcolors.append('y' if b[0] == 1 else 'b')

    xl, yl, thl = [], [], []
    carl = []
    dt_s = int(5)  # subsample factor for Gazebo
    for i in range(len(path)):
        xl.append(path[i].pos[0])
        yl.append(path[i].pos[1])
        thl.append(path[i].pos[2])
        carl.append(path[i].model[0])

    # Subsample: take every dt_s-th point and always include the last
    indices = list(range(0, len(xl), dt_s))
    if indices[-1] != len(xl) - 1:
        indices.append(len(xl) - 1)
    xs  = [xl[i]  for i in indices]
    ys  = [yl[i]  for i in indices]
    ths = [thl[i] for i in indices]

    # Infer direction per waypoint: positive dot product → forward, negative → reverse
        # Infer direction per waypoint: positive dot product → forward, negative → reverse
    dirs = []
    for k in range(len(xs) - 1):
        dx = xs[k+1] - xs[k]
        dy = ys[k+1] - ys[k]

        seg_len = sqrt(dx*dx + dy*dy)
        if seg_len < 1e-6:
            dirs.append(dirs[-1] if dirs else 1.0)
            continue

        avg_theta = atan2(
            sin(ths[k]) + sin(ths[k+1]),
            cos(ths[k]) + cos(ths[k+1])
        )

        fwd = cos(avg_theta) * dx + sin(avg_theta) * dy
        dirs.append(1.0 if fwd >= 0.0 else -1.0)

    dirs.append(dirs[-1] if dirs else 1.0)


    global WAYPOINTS
    WAYPOINTS = list(zip(xs, ys, ths, dirs))
    #print(WAYPOINTS)
    
    start_state = car.get_car_state(car.start_pos)
    end_state = car.get_car_state(car.end_pos)

    # plot and annimation
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, env.lx)
    ax.set_ylim(0, env.ly)
    ax.set_aspect("equal")

    if grid_on:
        ax.set_xticks(np.arange(0, env.lx, grid.cell_size))
        ax.set_yticks(np.arange(0, env.ly, grid.cell_size))
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
    """Obstacle map matching the Gazebo world (5.21m x 2.75m)."""

    def __init__(self):

        def box_to_rect(cx, cy, w, h):
            """Convert (center_x, center_y, width, height) to (x_min, y_min, w, h)."""
            return [cx - w/2, cy - h/2, w, h]

        self.obs = [

            # ==================== OUTER WALLS ====================

            # South wall left
            box_to_rect(1.65, 0.0, 3.3, 0.05),

            # South wall step (vertical)
            box_to_rect(3.3, 0.1, 0.05, 0.2),

            # South wall right
            box_to_rect(4.255, 0.2, 1.91, 0.05),

            # North wall
            box_to_rect(2.605, 2.75, 5.21, 0.05),

            # West wall bottom
            box_to_rect(0.0, 0.5, 0.05, 1.0),

            # West notch bottom
            box_to_rect(0.25, 1.0, 0.5, 0.05),

            # West notch right
            box_to_rect(0.5, 1.1, 0.05, 0.2),

            # West notch top
            box_to_rect(0.25, 1.2, 0.5, 0.05),

            # West wall top
            box_to_rect(0.0, 1.975, 0.05, 1.55),

            # East wall
            box_to_rect(5.21, 1.475, 0.05, 2.55),

            # ==================== INTERNAL OBSTACLES ====================

            # Obstacle 1
            box_to_rect(1.3, 1.85, 0.2, 0.4),

            # Obstacle 2
            box_to_rect(2.5, 1.85, 0.5, 0.4),

            # Obstacle 3
            box_to_rect(3.86, 1.9, 0.5, 0.2),

            # Obstacle 4  (shifted up to cy=0.8 so WP1 heading north has arc clearance)
            box_to_rect(1.6, 0.8, 0.5, 0.2),

            # Obstacle 5
            box_to_rect(3.41, 0.9, 0.5, 0.2),
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
        img_topic = "/camera/rgb/image_raw"
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
    file_source = '/home/miguel/catkin_ws/'
    file_destination = '/home/miguel/catkin_ws/src/assigment4_ttk4192/scripts'
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



    rospy.loginfo("Opening gripper")
    manip.open_gripper()
    rospy.sleep(1)

    rospy.loginfo("Closing gripper")
    manip.close_gripper()
    rospy.sleep(1)

def get_current_gazebo_pose(timeout=5.0):
    """Block until we get one ModelStates message; return (x, y, theta)."""
    msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=timeout)
    try:
        idx = msg.name.index(GAZEBO_MODEL_NAME)
    except ValueError:
        rospy.logerr("Model %s not found in /gazebo/model_states", GAZEBO_MODEL_NAME)
        return None
    pose = msg.pose[idx]
    quat = [pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w]
    (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
    return [pose.position.x, pose.position.y, yaw]


def move_robot_waypoint0_waypoint1(start_pos, end_pos, safe_dis=0.08):
    """Plan and execute a single leg using Hybrid A* with spline connector."""
    actual_pose = get_current_gazebo_pose()

    if actual_pose is not None:
        dx = actual_pose[0] - start_pos[0]
        dy = actual_pose[1] - start_pos[1]
        dist = sqrt(dx * dx + dy * dy)

        if dist < 0.20:
            # Close to nominal waypoint — use nominal position with actual heading
            # This avoids edge-case collisions from tiny offsets
            use_start = [start_pos[0], start_pos[1], actual_pose[2]]
            print(f"Robot near nominal WP (dist={dist:.3f}m) — snapping to nominal position")
        else:
            # Far from nominal — robot didn't reach target, use actual pose
            use_start = actual_pose
            print(f"Robot far from nominal WP (dist={dist:.3f}m) — using actual Gazebo pose")
    else:
        use_start = start_pos
        print("WARNING: Could not get Gazebo pose, using nominal start")

    print(f"Computing hybrid A* path (spline connector, safe_dis={safe_dis}m)")
    print("Start:", use_start)
    print("End:", end_pos)
    main_hybrid_a(heu=1, start_pos=use_start, end_pos=end_pos,
                  reverse=True, extra=False, grid_on=False, use_spline=True,
                  safe_dis=safe_dis)

    if not WAYPOINTS:
        print("WARNING: No path found — skipping this leg")
        return

    print("Executing path following")
    turtlebot_move(final_heading=end_pos[2])
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
        rospy.init_node('mission_planner', anonymous=False)   # <-- ADD THIS

        print()
        print("************ TTK4192 - Assigment 4 **************************")
        print()
        print("AI planners: GraphPlan")
        print("Path-finding: Hybrid A-star")
        print("GNC Controller: PID path-following")
        print("Robot: Turtlebot3 waffle-pi")
        print("date: 20.03.23")
        print()
        print("**************************************************************")
        print()
 
        # WAYPOINTS = [[1.99,2.45],[2.99,2.45],]
        # turtlebot_move()
        # check_seals_valve_picture_eo(1)
        # use_gripper_exe()

        plan_file = "/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning/tmp_sas_plan.1"
        plan = parse_plan(plan_file)
        print("\n[Planner] Loaded existing plan:")
        for step, a in enumerate(plan, 1):
            print(f"  {step}: {a['action']} {' '.join(a['args'])}")

        # 5.3) Start mission execution
        print("")
        print("Starting mission execution")
        
        # Start simulations with battery = 100%
        battery = 100
        task_finished = 0
        task_total = len(plan)
        i_ini = 0
        # Obstacle safety margin — increased after WP2→WP5 to keep extra clearance
        obstacle_margin = 0.02

        while i_ini < task_total:
            action = plan[i_ini]
            action_name = action['action']
            action_args = action['args']

            print(f"\n[Mission] Step {i_ini+1}/{task_total}: {action_name} {' '.join(action_args)}")

            if action_name == "move_robot":
                # args: [robot, from_wp, to_wp, route]
                from_wp = action_args[1]
                to_wp   = action_args[2]
                print(f"[Mission] Moving robot from {from_wp} to {to_wp}")
                move_robot_waypoint0_waypoint1(WP_MAP[from_wp], WP_MAP[to_wp],
                                               safe_dis=obstacle_margin)
                # After completing WP2→WP5 use a larger margin for remaining legs
                if from_wp == 'waypoint2' and to_wp == 'waypoint5':
                    obstacle_margin = 0.08
                    print(f"[Mission] Obstacle margin increased to {obstacle_margin}m")
                stop_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
                stop_pub.publish(Twist())
                rospy.sleep(1.0)

            elif action_name == "take_picture_pump_ir":
                # args: [robot, waypoint, camera,  pump]
                waypoint = action_args[1] if len(action_args) > 1 else "unknown"
                target   = action_args[3] if len(action_args) > 2 else "unknown"
                print(f"[Mission] Inspecting pump at {waypoint} — target: {target}")
                # taking_photo_exe()  # uncomment when ready
                time.sleep(1)

            elif action_name == "check_seals_valve_picture_eo":
                # args: [robot, waypoint, camera, valve]
                waypoint = action_args[1] if len(action_args) > 1 else "unknown"
                target   = action_args[2] if len(action_args) > 2 else "unknown"
                print(f"[Mission] Checking valve EO at {waypoint} — target: {target}")
                # taking_photo_exe()  # uncomment when ready
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