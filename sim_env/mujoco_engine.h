#ifndef WRAPPER_MUJOCO_ENGINE_H
#define WRAPPER_MUJOCO_ENGINE_H

/**
 *
 */
class Sim {

public:
    /**
     * Initializes an instance of MuJoCo environment.
     *
     * @param model_path the path of the xml file of the model
     * @param rendering  true if rendering with GUI, false otherwise
     */
    explicit Sim(const char *model_path, bool rendering);

    /**
     * Destructs the current instance of MuJoCo simulation environment.
     */
    void destruct();

    /**
    * Resets the instance of the MuJoCo simulation with all given initial configurations.
    *
    * @param obstacle_pos_x the initial position x of the obstacle
    * @param obstacle_pos_y the initial position y of the obstacle
    * @param agent_pos_x    the initial position x of the agent
    * @param agent_pos_y    the initial position y of the agent
    */
    void reset(double obstacle_pos_x, double obstacle_pos_y, double agent_pos_x, double agent_pos_y);

    /**
     * Gets the index of a body using its name.
     *
     * @param body_name the name of the body
     *
     * @return the index of the body
     */
    int get_body_index(const char *body_name);

    /**
     * Gets the index of a joint using its name.
     *
     * @param jnt_name the name of the joint
     *
     * @return the index of the joint
     */
    int get_jnt_index(const char *jnt_name);

    /**
     * Gets the initial position x of the simulation object represented by the given index.
     *
     * @param body_index the index representing a simulation object
     *
     * @return the initial position x of the simulation object
     */
    double get_body_ini_pos_x(int body_index);

    /**
     * Gets the initial position y of the simulation object represented by the given index.
     *
     * @param body_index the index representing a simulation object
     *
     * @return the initial position y of the simulation object
     */
    double get_body_ini_pos_y(int body_index);

    /**
     * Gets the reference position of the joint represented by the given index.
     *
     * @param jnt_index the index representing a joint
     *
     * @return the reference position of the joint
     */
    double get_qpos(int jnt_index);

    /**
     * Gets the absolute position x of the simulation object represented by the given index.
     *
     * @param body_index the index representing a simulation object
     *
     * @return the initial position x of the simulation object
     */
    double get_xpos_x(int body_index);

    /**
     * Gets the absolute position y of the simulation object represented by the given index.
     *
     * @param body_index the index representing a simulation object
     *
     * @return the initial position y of the simulation object
     */
    double get_xpos_y(int body_index);

    /**
     * Renders the simulation.
     */
    void render();

    /**
     * Steps the simulation.
     *
     * @param obstacle_speed the speed of the obstacle to be set
     * @param agent_x_speed  the x speed of the agent to be set
     * @param agent_y_speed  the x speed of the agent to be set
     */
    void step(double obstacle_speed, double agent_x_speed, double agent_y_speed);

private:
    /**
     * Initializes the MuJoCo environment.
     *
     * @param modelPath the path of the xml file of the model
     */
    void init_mujoco(const char *modelPath);

    /**
     * Initializes the rendering configuration.
     *
     * @param cam_distance  camera distance
     * @param cam_azimuth   camera azimuth
     * @param cam_elevation camera elevation
     */
    void init_render(int cam_distance, int cam_azimuth, int cam_elevation);
};

#endif //WRAPPER_MUJOCO_ENGINE_H