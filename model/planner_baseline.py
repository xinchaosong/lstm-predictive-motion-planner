import random
from math import sqrt
from collections import namedtuple, deque

from model.planner import Area, choose_direction


def planner_baseline():
    map_size = 12
    step_resolution = 0.25
    step_size = 10
    obstacle_width = 10
    obstacle_thickness = 5
    agent_step = step_resolution * step_size
    Node = namedtuple('Node', ['x', 'y'])

    def is_valid_move(start_node, dir_x, dir_y):
        global obstacle
        end_x = start_node.x
        end_y = start_node.y
        start_step = calculate_step(start_node)

        for s in range(step_size):
            if (end_x < -map_size) or (end_x > map_size) or (end_y < -map_size) or (end_y > map_size):
                return False

            for obstacle in walls:
                if obstacle.interfere_node(end_x, end_y):
                    return False

            if obstacle_range.interfere_node(end_x, end_y):
                cur_step = start_step + s

                if cur_step >= 100:
                    return False

            end_x += dir_x * step_resolution
            end_y += dir_y * step_resolution

        return True

    def bfs(end_node):
        parents = {}
        finished = set()
        queue = deque()
        queue.append(origin)

        while queue:
            parent = queue.popleft()
            finished.add(parent)

            for child in route_dict[parent]:
                if child not in finished:
                    if child not in queue:
                        queue.append(child)
                        parents[child] = parent

        cur_node = end_node
        path = [end_node]

        while cur_node != origin:
            cur_node = parents[cur_node]
            path.append(cur_node)

        path.reverse()

        return path

    def calculate_step(end_node):
        path = bfs(end_node=end_node)

        return (len(path) - 1) * step_size

    obstacle = Area(0, 0, obstacle_width, obstacle_thickness)
    wall0 = Area(-8, 7.5, 12, 5)
    wall1 = Area(8, -7.5, 12, 5)
    walls = [wall0, wall1]
    target = Area(-8, 11, 8, 2)
    obstacle_range = Area(0, 0, map_size * 2 + 4, 5)
    origin = Node(2, -11)
    route_dict = {origin: set()}

    done = False
    last_node = None

    while not done:
        pos_rand_x = random.uniform(-map_size, map_size)
        pos_rand_y = random.uniform(-map_size, map_size)

        min_distance = float('inf')
        n_nearest = None

        for node in route_dict.keys():
            distance = sqrt(abs(node.x - pos_rand_x) + abs(node.y - pos_rand_y))

            if min_distance > distance:
                min_distance = distance
                n_nearest = node

        if min_distance > agent_step:
            continue

        direction_x, direction_y = choose_direction(n_nearest.x, n_nearest.y, pos_rand_x, pos_rand_y)

        if not is_valid_move(n_nearest, direction_x, direction_y):
            continue

        new_node_x = n_nearest.x + direction_x * agent_step
        new_node_y = n_nearest.y + direction_y * agent_step

        new_node = Node(new_node_x, new_node_y)

        if new_node in route_dict:
            continue

        route_dict[new_node] = set()
        route_dict[n_nearest].add(new_node)
        route_dict[new_node].add(n_nearest)

        for p in route_dict.keys():
            direction_x, direction_y = choose_direction(p.x, p.y, new_node.x, new_node.y)

            if abs(p.x - new_node.x) == agent_step and abs(p.y - new_node.y) == agent_step \
                    and is_valid_move(p, direction_x, direction_y):
                route_dict[p].add(new_node)

        if target.interfere_node(new_node_x, new_node_y):
            last_node = new_node
            done = True

    path = bfs(end_node=last_node)

    policy = []

    for i in range(len(path) - 1):
        if path[i + 1].x < path[i].x:
            policy_unit_x = -1
        elif path[i + 1].x > path[i].x:
            policy_unit_x = 1
        else:
            policy_unit_x = 0

        if path[i + 1].y < path[i].y:
            policy_unit_y = -1
        elif path[i + 1].y > path[i].y:
            policy_unit_y = 1
        else:
            policy_unit_y = 0

        policy.append([policy_unit_x, policy_unit_y])

    return policy
