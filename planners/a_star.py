import numpy as np
import heapq
import math

class AStarPlanner:
    def __init__(self, ox, oy, resolution, robot_radius):
        """
        ox: x coordinates of obstacles
        oy: y coordinates of obstacles
        resolution: grid size [m]
        robot_radius: collision buffer [m]
        """
        self.resolution = resolution
        self.rr = robot_radius
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.x_width, self.y_width = 0, 0
        self.obstacle_map = None
        
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # grid index
            self.y = y  # grid index
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy):
        """ Returns path: [[x, y], ...] """
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        
        # Priority Queue for efficiency
        pq = [] 
        heapq.heappush(pq, (0, self.calc_grid_index(start_node)))

        while True:
            if not open_set:
                print("Error: No path found")
                return None

            _, c_id = heapq.heappop(pq)
            current = open_set[c_id]

            # Check goal
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            # Expand search
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x, current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_grid_index(node)

                if n_id in closed_set: continue
                if not self.verify_node(node): continue

                if n_id not in open_set:
                    open_set[n_id] = node
                    # Heuristic: Euclidean distance
                    heuristic = math.hypot(node.x - goal_node.x, node.y - goal_node.y)
                    heapq.heappush(pq, (node.cost + heuristic, n_id))
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        return self.calc_final_path(goal_node, closed_set)

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return np.array([rx, ry]).T[::-1] # Reverse to get Start -> Goal

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    def verify_node(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def calc_grid_position(self, index, min_pos):
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    @staticmethod
    def get_motion_model():
        # dx, dy, cost (8-connected grid)
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]