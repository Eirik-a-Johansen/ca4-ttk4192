import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import matplotlib.animation as animation
from math import pi, tan, sqrt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle
from itertools import product
import argparse
from utils.grid import Grid
from utils.car import SimpleCar, State
from utils.environment import Environment
from utils.dubins_path import DubinsPath
from utils.astar import Astar
from utils.utils import plot_a_car, get_discretized_thetas, round_theta, same_point


""" ---------------------------------------------------------------
 Code for pathfinding using hybrid A-star 
 dependencies: Curve generator (Dubins path)
 Based on: Open source scripts in GitHub
 Date: 18.01.23
--------------------------------------------------------------------
"""

class HybridAstar:
    """ Hybrid A* search procedure. """
    def __init__(self, car, grid, reverse, unit_theta=pi/12, dt=1e-2, check_dubins=1,
                 epsilon=8.5, max_iter=5000):
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

        self.dubins = DubinsPath(self.car)
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

            # check safety of route-----------------------
            if phi == 0:
                pos1 = node.pos if m == 1 else pos
                pos2 = pos if m == 1 else node.pos
                safe = self.dubins.is_straight_route_safe(pos1, pos2)
            else:
                if m == 1:
                    # forward: centre and direction come directly from node position
                    d, c, r = self.car.get_params(node.pos, phi)
                    safe = self.dubins.is_turning_route_safe(node.pos, pos, d, c, r)
                else:
                    # reverse: same geometric centre as the equivalent forward arc,
                    # but the effective turning direction is flipped
                    _, c, r = self.car.get_params(node.pos, phi)
                    d = -1 if phi > 0 else 1
                    safe = self.dubins.is_turning_route_safe(node.pos, pos, d, c, r)
            # --------------------------------------------

            if not safe:
                continue

            child = self.construct_node(pos)
            child.phi = phi
            child.m = m
            child.parent = node
            child.g = node.g + self.arc
            child.g_ = node.g_ + self.arc

            # direction-change penalty always applied (not just with --extra)
            # so the planner only switches gears when it genuinely helps
            if node.m and node.m != m:
                child.g += self.w5 * self.arc

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
    
    def smooth_path(self, path, window_size=5, method='moving_average'):
        """ Smooth the path using different smoothing methods. """
        if len(path) < window_size:
            return path
        
        if method == 'moving_average':
            return self._smooth_moving_average(path, window_size)
        elif method == 'weighted_average':
            return self._smooth_weighted_average(path, window_size)
        else:
            return path
    
    def _smooth_moving_average(self, path, window_size=5):
        """ Smooth the path using simple moving average filter. """
        smoothed_path = []
        
        # Keep the first point unchanged
        smoothed_path.append(path[0])
        
        # Apply moving average to intermediate points
        for i in range(1, len(path) - 1):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)
            
            # Calculate average position
            avg_x = sum(p.pos[0] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            avg_y = sum(p.pos[1] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            avg_theta = path[i].pos[2]  # Keep original orientation
            
            # Create new car state with smoothed position
            smoothed_state = State([avg_x, avg_y, avg_theta], path[i].model)
            smoothed_path.append(smoothed_state)
        
        # Keep the last point unchanged
        smoothed_path.append(path[-1])
        
        return smoothed_path
    
    def _smooth_weighted_average(self, path, window_size=5):
        """ Smooth the path using weighted moving average (more weight to center). """
        smoothed_path = []
        
        # Keep the first point unchanged
        smoothed_path.append(path[0])
        
        # Create weights (gaussian-like distribution)
        weights = np.exp(-np.linspace(-2, 2, window_size)**2)
        weights = weights / np.sum(weights)
        
        # Apply weighted average to intermediate points
        for i in range(1, len(path) - 1):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)
            
            # Get points in window
            window_points = path[start_idx:end_idx]
            window_weights = weights[len(weights)//2 - (i - start_idx): len(weights)//2 + (end_idx - i)]
            
            # Calculate weighted average position
            avg_x = sum(p.pos[0] * w for p, w in zip(window_points, window_weights)) / sum(window_weights)
            avg_y = sum(p.pos[1] * w for p, w in zip(window_points, window_weights)) / sum(window_weights)
            avg_theta = path[i].pos[2]  # Keep original orientation
            
            # Create new car state with smoothed position
            smoothed_state = State([avg_x, avg_y, avg_theta], path[i].model)
            smoothed_path.append(smoothed_state)
        
        # Keep the last point unchanged
        smoothed_path.append(path[-1])
        
        return smoothed_path
    
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


def main_hybrid_a(heu, start_pos, end_pos, reverse, extra, grid_on, smooth=True, smooth_method='weighted_average'):

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
    
    # Apply path smoothing
    if smooth:
        path = hastar.smooth_path(path, method=smooth_method)
        print('Path smoothed with {} filter'.format(smooth_method))
    
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
    xl_np=xl_np-10
    yl_np=np.array(yl_np1)
    yl_np=yl_np-10
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
    """Obstacle map matching the Gazebo world (5.21m x 2.55m)."""

    def __init__(self):

        # Start and goal (you can change these as needed)
        self.start_pos2 = [4, 4, 0]
        self.end_pos2   = [4, 8, -pi]

        # Helper: convert from (center_x, center_y, width, height)
        # to (x_min, y_min, width, height)
        def box_to_rect(cx, cy, w, h):
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
            box_to_rect(2.5, 1.85, 0.4, 0.4),

            # Obstacle 3
            box_to_rect(3.86, 1.9, 0.4, 0.2),

            # Obstacle 4
            box_to_rect(1.6, 0.7, 0.5, 0.2),

            # Obstacle 5
            box_to_rect(3.41, 0.9, 0.5, 0.2),
        ]

if __name__ == '__main__':
    print("Executing hybrid A* algorithm")
    p = argparse.ArgumentParser()
    p.add_argument('-heu', type=int, default=1, help='heuristic type')
    p.add_argument('-r', action='store_true', help='allow reverse or not')
    p.add_argument('-e', action='store_true', help='add extra cost or not')
    p.add_argument('-g', action='store_true', help='show grid or not')
    p.add_argument('-s', action='store_true', help='enable path smoothing')
    p.add_argument('-sm', type=str, default='weighted_average', choices=['moving_average', 'weighted_average'], help='smoothing method')
    args = p.parse_args()

    WP_MAP = {
        'WP0': [0.6,  0.3,  0],   # valid as-is
        'WP1': [1.6,  0.3,  np.pi/2],   # valid as-is
        'WP2': [2.9,  1.3,  -np.pi/2],   # was [3.41,1.0]: inside box obstacle at x=3.06-3.76, y=0.7-1.1
        'WP3': [3.36,  2.7,  np.pi/2],   # was [3.41,1.0]: inside box obstacle at x=3.06-3.76, y=0.7-1.1
        'WP4': [4.5,  0.5,  0],   # was [5.15,0.25]: too close to right wall and floor step
        'WP5': [0.87, 2.4,  0],   # was [0.87,2.56]: car top edge clipped top-wall safe zone
        'WP6': [4.3,  2.2,  np.pi/2],   # was [3.86,1.8]: inside box obstacle at x=3.56-4.16, y=1.7-2.1
    }

    mission = ['WP2', 'WP4', 'WP1', 'WP2', 'WP5', 'WP6']
    #mission = ['WP1', 'WP0']
    for i in range(len(mission) - 1):
        start_name = mission[i]
        end_name   = mission[i + 1]
        start_pos  = WP_MAP[start_name]
        end_pos    = WP_MAP[end_name]
        print(f"\n--- Leg {i+1}: {start_name} -> {end_name} ---")
        main_hybrid_a(args.heu, start_pos, end_pos, args.r, args.e, args.g, smooth=args.s, smooth_method=args.sm)
        print(f"Leg {i+1} done.")
