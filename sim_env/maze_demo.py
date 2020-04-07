from sim_env.env import SimEnv

env = SimEnv(xml_name="maze.xml", recompile_cpp=False, rendering=True)
env.reset(obstacle_pos=(-5, 0), agent_pos=(2, -11.0))

while True:
    env.render()
