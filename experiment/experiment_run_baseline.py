import random
import time
from multiprocessing import Process, Manager

from model.planner_baseline import planner_baseline
from sim_env.env import SimEnv

env = SimEnv(xml_name="maze.xml", recompile_cpp=True, rendering=True)
x_limit = 10
x_start = random.uniform(-x_limit, x_limit)
obstacle_speed = random.choice([-1, 1])
env.reset(obstacle_pos=(x_start, 0), agent_pos=(2, -11.0))
done = False
idx = 0
step = 0

obs = []

for i in range(50):
    env.render()
    obstacle_x = env.get_xpos(env.obstacle_body_index)
    obs.append(obstacle_x[0])

    env.step([obstacle_speed, 0, 0])

    if obstacle_x[0] < -10:
        obstacle_speed = 1

    if obstacle_x[0] > 10:
        obstacle_speed = -1


def task(policy_shared):
    temp_p = None

    while temp_p is None:
        temp_p = planner_baseline()

    policy_shared += temp_p
    switch.value = 0


switch = Manager().Value('i', 1)
policy = Manager().list()
cal = Process(target=task, args=(policy,))
cal.start()

while switch.value == 1:
    env.render()

print(policy)

while not done:
    env.render()
    obstacle_x = env.get_xpos(env.obstacle_body_index)
    agent_x = env.get_xpos(env.agent_body_index)

    if step >= len(policy):
        print("Failed.")
        break

    env.step([obstacle_speed, policy[step][0], policy[step][1]])

    if obstacle_x[0] < -10:
        obstacle_speed = 1

    if obstacle_x[0] > 10:
        obstacle_speed = -1

    idx += 1

    if idx % 10 == 0:
        step += 1

        if agent_x[0] < -4 and agent_x[1] > 10:
            done = True

    time.sleep(0.1)
else:
    print("Succeed!")

env.close()
