import os
import pathlib
import sys
import ctypes


class SimEnv:
    """
    The simulation environment based on MuJoCo 200 and the corresponding c++ control engine.
    """

    def __init__(self, xml_name, recompile_cpp=False, rendering=True):
        """
        The initialization of the class.
        """
        if recompile_cpp:
            self._update_wrapper()

        if sys.platform.startswith('darwin'):
            cdll_path = os.path.join(os.path.dirname(__file__), "libsimenv.dylib")
        elif sys.platform.startswith('linux'):
            cdll_path = os.path.join(os.path.dirname(__file__), "libsimenv.so")
        elif sys.platform.startswith('win32'):
            cdll_path = os.path.join(os.path.dirname(__file__), "libsimenv.dll")
        else:
            raise EnvironmentError("Unknown operating system found.")

        model_path = os.path.join(pathlib.Path(__file__).parent, "mujoco_model/", xml_name).encode('utf-8')
        self.rendering = rendering

        # C++ control engine.
        self.wrapper = ctypes.CDLL(cdll_path)
        self.instance = self.wrapper.get_instance(ctypes.c_char_p(model_path), ctypes.c_bool(rendering))

        # Indices of the object bodies.
        self.obstacle_body_index = self.get_body_index("obstacle")
        self.agent_body_index = self.get_body_index("agent")

        # Indices of the joints.
        self.obstacle_jnt_index = self.get_jnt_index("slider:obstacle")
        self.agent_jnt_x_index = self.get_jnt_index("slider:agent-obstacle_x")
        self.agent_jnt_y_index = self.get_jnt_index("slider:agent-y")

        # Initial positions from the configuration.
        self.obstacle_pos = self.get_body_ini_pos(self.obstacle_body_index)
        self.agent_pos = self.get_body_ini_pos(self.agent_body_index)

    def close(self):
        """
        Closes the instance.

        :return: None
        """
        self.wrapper.close(self.instance)

    def reset(self, obstacle_pos=(0.0, 0.0), agent_pos=(2.0, -11.0)):
        """
        Resets the MuJoCo model to the initial state.

        :return: the observations of the environment
        """

        obstacle_pos_x_double = ctypes.c_double(obstacle_pos[0])
        obstacle_pos_y_double = ctypes.c_double(obstacle_pos[1])
        agent_pos_x_double = ctypes.c_double(agent_pos[0])
        agent_pos_y_double = ctypes.c_double(agent_pos[1])

        self.wrapper.reset(self.instance, obstacle_pos_x_double, obstacle_pos_y_double,
                           agent_pos_x_double, agent_pos_y_double)
        self.obstacle_pos = self.get_body_ini_pos(self.obstacle_body_index)
        self.agent_pos = self.get_body_ini_pos(self.agent_body_index)

    def render(self):
        """
        Renders the environment.

        :return: None
        """
        if self.rendering:
            self.wrapper.render(self.instance)

    def step(self, speed):
        """
        Makes the actuator of the MuJoCo model do the given action and returns the observations.

        :param speed: a tuple representing the speed that needs to be executed
        """

        obstacle_speed_double = ctypes.c_double(speed[0])
        agent_x_speed_double = ctypes.c_double(speed[1])
        agent_y_speed_double = ctypes.c_double(speed[2])

        self.wrapper.step(self.instance, obstacle_speed_double, agent_x_speed_double, agent_y_speed_double)

    def get_body_index(self, body_name):
        """
        Gets the index of a body using its name.

        :param body_name: the name of the body
        :return: the index of the body
        """
        return self.wrapper.get_body_index(self.instance, body_name.encode('utf-8'))

    def get_jnt_index(self, jnt_name):
        """
        Gets the index of a joint using its name.

        :param jnt_name: the name of the joint
        :return: the index of the joint
        """
        return self.wrapper.get_jnt_index(self.instance, jnt_name.encode('utf-8'))

    def get_body_ini_pos(self, body_index):
        """
        Gets the initial position of the simulation object represented by the given index.

        :param body_index: the index representing a simulation object
        :return: the initial position of the simulation object
        """
        pos_x_func = self.wrapper.get_body_ini_pos_x
        pos_y_func = self.wrapper.get_body_ini_pos_y
        pos_x_func.restype = ctypes.c_double
        pos_y_func.restype = ctypes.c_double
        pos_x = pos_x_func(self.instance, body_index)
        pos_y = pos_y_func(self.instance, body_index)

        return pos_x, pos_y

    def get_qpos(self, jnt_index):
        """
        Gets the reference position of the joint represented by the given index.

        :param jnt_index: the index representing a joint
        :return: the reference position of the joint
        """
        func = self.wrapper.get_qpos
        func.restype = ctypes.c_double

        return func(self.instance, jnt_index)

    def get_xpos(self, body_index):
        """
        Gets the initial position of the simulation object represented by the given index.

        :param body_index: the index representing a simulation object
        :return: the initial position of the simulation object
        """
        xpos_x_func = self.wrapper.get_xpos_x
        xpos_y_func = self.wrapper.get_xpos_y
        xpos_x_func.restype = ctypes.c_double
        xpos_y_func.restype = ctypes.c_double
        xpos_x = xpos_x_func(self.instance, body_index)
        xpos_y = xpos_y_func(self.instance, body_index)

        return xpos_x, xpos_y

    @staticmethod
    def _update_wrapper():
        """
        Updates wrapper.dylib.

        :return: None
        """
        sh_path = os.path.join(os.path.dirname(__file__), "build_wrapper.sh")
        print(os.system(sh_path))
