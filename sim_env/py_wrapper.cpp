#include <new>
#include "mujoco_engine.h"

extern "C"
{
/**
 * Creates and returns a new instance of the MuJoCo simulation.
 *
 * @param model_path the path of the xml file of the model
 * @param rendering  true if rendering with GUI, false otherwise
 *
 * @return a pointer to the generated instance
 */
void *get_instance(const char *model_path, bool rendering) {
    return new(std::nothrow) Sim(model_path, rendering);
}

/**
 * Closes a new instance of the MuJoCo simulation.
 *
 * @param ptr the pointer to the current instance
 */
void close(void *ptr) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    ref->destruct();
}

/**
 * Resets the instance of the MuJoCo simulation with all given initial configurations.
 *
 * @param ptr            the pointer to the current instance
 * @param obstacle_pos_x the initial position x of the obstacle
 * @param obstacle_pos_y the initial position y of the obstacle
 * @param agent_pos_x    the initial position x of the agent
 * @param agent_pos_y    the initial position y of the agent
 */
void reset(void *ptr, double obstacle_pos_x, double obstacle_pos_y, double agent_pos_x, double agent_pos_y) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    ref->reset(obstacle_pos_x, obstacle_pos_y, agent_pos_x, agent_pos_y);
}

/**
 * Gets the index of a body using its name.
 *
 * @param ptr       the pointer to the current instance
 * @param body_name the name of the body
 *
 * @return the index of the body
 */
int get_body_index(void *ptr, const char *body_name) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_body_index(body_name);
}

/**
 * Gets the index of a joint using its name.
 *
 * @param ptr      the pointer to the current instance
 * @param jnt_name the name of the joint
 *
 * @return the index of the joint
 */
int get_jnt_index(void *ptr, const char *jnt_name) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_jnt_index(jnt_name);
}

/**
 * Gets the initial position x of the simulation object represented by the given index.
 *
 * @param ptr        the pointer to the current instance
 * @param body_index the index representing a simulation object
 *
 * @return the initial position x of the simulation object
 */
double get_body_ini_pos_x(void *ptr, int body_index) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_body_ini_pos_x(body_index);
}

/**
 * Gets the initial position y of the simulation object represented by the given index.
 *
 * @param ptr        the pointer to the current instance
 * @param body_index the index representing a simulation object
 *
 * @return the initial position y of the simulation object
 */
double get_body_ini_pos_y(void *ptr, int body_index) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_body_ini_pos_y(body_index);
}

/**
 * Gets the reference position of the joint represented by the given index.
 *
 * @param ptr       the pointer to the current instance
 * @param jnt_index the index representing a joint
 *
 * @return the reference position of the joint
 */
double get_qpos(void *ptr, int jnt_index) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_qpos(jnt_index);
}

/**
 * Gets the absolute position x of the simulation object represented by the given index.
 *
 * @param ptr        the pointer to the current instance
 * @param body_index the index representing a simulation object
 *
 * @return the initial position x of the simulation object
 */
double get_xpos_x(void *ptr, int body_index) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_xpos_x(body_index);
}

/**
 * Gets the absolute position y of the simulation object represented by the given index.
 *
 * @param ptr        the pointer to the current instance
 * @param body_index the index representing a simulation object
 *
 * @return the initial position y of the simulation object
 */
double get_xpos_y(void *ptr, int body_index) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    return ref->get_xpos_y(body_index);
}

/**
 * Renders the simulation.
 *
 * @param ptr the pointer to the current instance
 */
void render(void *ptr) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    ref->render();
}

/**
 * Steps the simulation.
 *
 * @param ptr            the pointer to the current instance
 * @param obstacle_speed the speed of the obstacle to be set
 * @param agent_x_speed  the x speed of the agent to be set
 * @param agent_y_speed  the x speed of the agent to be set
 */
void step(void *ptr, double obstacle_speed, double agent_x_speed, double agent_y_speed) {
    Sim *ref = reinterpret_cast<Sim *>(ptr);
    ref->step(obstacle_speed, agent_x_speed, agent_y_speed);
}
}
