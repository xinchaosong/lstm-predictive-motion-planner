#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "mujoco.h"
#include "glfw3.h"
#include "mujoco_engine.h"


// MuJoCo data structures
mjModel *m = nullptr;                  // MuJoCo model
mjData *d = nullptr;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
GLFWwindow *window = nullptr;

// Mouse interaction.
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// Flags.
bool paused = true;

// Indices.
int slider_obstacle_jnt_idx;
int slider_agent_jnt_x_idx;
int slider_agent_jnt_y_idx;

int obstacle_body_idx;
int agent_body_idx;

// Keyboard callback.
void keyboard(GLFWwindow *glfwWindow, int key, int scancode, int act, int mods) {
    // backspace: reset_default simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_SPACE) {
        paused = !paused;
    }

    if (0) {
        opt.frame = mjFRAME_WORLD; // Show the world frame
    }
}


// Mouse button callback.
void mouse_button(GLFWwindow *glfwWindow, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // Updates mouse position.
    glfwGetCursorPos(window, &lastx, &lasty);
}


// Mouse move callback.
void mouse_move(GLFWwindow *glfwWindow, double pos_x, double pos_y) {
    // No buttons down: nothing to do.
    if (!button_left && !button_middle && !button_right)
        return;

    // Computes mouse displacement, save.
    double dx = pos_x - lastx;
    double dy = pos_y - lasty;
    lastx = pos_x;
    lasty = pos_y;

    // Gets current window size.
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Gets shift key state.
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // Determines action based on mouse button.
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // Moves camera.
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}


// Scroll callback.
void scroll(GLFWwindow *glfwWindow, double offset_x, double offset_y) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * offset_y, &scn, &cam);
}

/**
 * Initializes an instance of MuJoCo environment.
 *
 * @param model_path the path of the xml file of the model
 * @param rendering  true if rendering with GUI, false otherwise
 */
Sim::Sim(const char *model_path, bool rendering) {
    this->init_mujoco(model_path);

    obstacle_body_idx = this->get_body_index("obstacle");
    agent_body_idx = this->get_body_index("agent");

    slider_obstacle_jnt_idx = this->get_jnt_index("slider:obstacle");
    slider_agent_jnt_x_idx = this->get_jnt_index("slider:agent-x");
    slider_agent_jnt_y_idx = this->get_jnt_index("slider:agent-y");

    if (rendering) {
        this->init_render(45, 90, -80);
    }
}

/**
 * Destructs the current instance of MuJoCo simulation environment.
 */
void Sim::destruct() {
    // Frees visualization storage.
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Frees MuJoCo model and data, deactivate.
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // Terminates GLFW (crashes with Linux NVidia drivers).
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
}

/**
 * Resets the instance of the MuJoCo simulation with all given initial configurations.
 *
 * @param obstacle_pos_x the initial position x of the obstacle
 * @param obstacle_pos_y the initial position y of the obstacle
 * @param agent_pos_x    the initial position x of the agent
 * @param agent_pos_y    the initial position y of the agent
 */
void Sim::reset(double obstacle_pos_x, double obstacle_pos_y, double agent_pos_x, double agent_pos_y) {

    mj_resetData(m, d);

    m->body_pos[3 * obstacle_body_idx] = obstacle_pos_x;
    m->body_pos[3 * obstacle_body_idx + 1] = obstacle_pos_y;
    m->body_pos[3 * agent_body_idx] = agent_pos_x;
    m->body_pos[3 * agent_body_idx + 1] = agent_pos_y;

    d->qpos[slider_obstacle_jnt_idx] = 0;
    d->qpos[slider_agent_jnt_x_idx] = 0;
    d->qpos[slider_agent_jnt_y_idx] = 0;

    mj_forward(m, d);
}

/**
 * Gets the index of a body using its name.
 *
 * @param body_name the name of the body
 *
 * @return the index of the body
 */
int Sim::get_body_index(const char *body_name) {
    return mj_name2id(m, mjOBJ_BODY, body_name);
}


/**
 * Gets the index of a joint using its name.
 *
 * @param jnt_name the name of the joint
 *
 * @return the index of the joint
 */
int Sim::get_jnt_index(const char *jnt_name) {
    return mj_name2id(m, mjOBJ_JOINT, jnt_name);
}

/**
 * Gets the initial position x of the simulation object represented by the given index.
 *
 * @param body_index the index representing a simulation object
 *
 * @return the initial position x of the simulation object
 */
double Sim::get_body_ini_pos_x(int body_index) {
    return m->body_pos[3 * body_index];
}

/**
 * Gets the initial position y of the simulation object represented by the given index.
 *
 * @param body_index the index representing a simulation object
 *
 * @return the initial position y of the simulation object
 */
double Sim::get_body_ini_pos_y(int body_index) {
    return m->body_pos[3 * body_index + 1];
}

/**
 * Gets the reference position of the joint represented by the given index.
 *
 * @param jnt_index the index representing a joint
 *
 * @return the reference position of the joint
 */
double Sim::get_qpos(int jnt_index) {
    return d->qpos[jnt_index];
}

/**
 * Gets the absolute position x of the simulation object represented by the given index.
 *
 * @param body_index the index representing a simulation object
 *
 * @return the initial position x of the simulation object
 */
double Sim::get_xpos_x(int body_index) {
    return d->xpos[3 * body_index];
}

/**
 * Gets the absolute position y of the simulation object represented by the given index.
 *
 * @param body_index the index representing a simulation object
 *
 * @return the initial position y of the simulation object
 */
double Sim::get_xpos_y(int body_index) {
    return d->xpos[3 * body_index + 1];
}

/**
* Renders the simulation.
*/
void Sim::render() {
    // Gets framebuffer viewport.
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // Updates scene and render.
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // Info during rendering.
    char status[1000] = "";
    sprintf(status, "%-5.4f\n%d", d->time, d->ncon);
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport, "Time\nContacts", status, &con);

    // Swaps OpenGL buffers (blocking call due to v-sync).
    glfwSwapBuffers(window);

    // Processes pending GUI events, call GLFW callbacks.
    glfwPollEvents();
}

/**
  * Steps the simulation.
  *
  * @param obstacle_speed the speed of the obstacle to be set
  * @param agent_x_speed  the x speed of the agent to be set
  * @param agent_y_speed  the x speed of the agent to be set
  */
void Sim::step(double obstacle_speed, double agent_x_speed, double agent_y_speed) {
    mjtNum simstart = d->time;

    while (d->time - simstart < 0.5) {
        mj_step1(m, d);
        d->ctrl[0] = (mjtNum) obstacle_speed;
        d->ctrl[1] = (mjtNum) agent_x_speed;
        d->ctrl[2] = (mjtNum) agent_y_speed;
        mj_step2(m, d);
    }
}

/**
 * Initializes the MuJoCo environment.
 *
 * @param modelPath the path of the xml file of the model
 */
void Sim::init_mujoco(const char *modelPath) {
    // Activates software.
    const char *mjKeyPath = std::getenv("MUJOCO_KEY");
    mj_activate(mjKeyPath);

    //std::cout << modelPath << std::endl;
    // Loads and compiles model.
    m = mj_loadXML(modelPath, nullptr, nullptr, 0);
    if (!m) mju_error("Model cannot be loaded");

    // Makes data.
    d = mj_makeData(m);
}

/**
 * Initializes the rendering configuration.
 *
 * @param cam_distance  camera distance
 * @param cam_azimuth   camera azimuth
 * @param cam_elevation camera elevation
 */
void Sim::init_render(int cam_distance, int cam_azimuth, int cam_elevation) {
    // Initialize GLFW.
    if (!glfwInit())
        mju_error("Could not initialize GLFW");

    // Creates window, make OpenGL context current, request v-sync.
    window = glfwCreateWindow(1200, 900, "", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize visualization data structures.
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // Creates scene and context.
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Installs GLFW mouse and keyboard callbacks.
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    cam.distance = cam_distance;
    cam.azimuth = cam_azimuth;
    cam.elevation = cam_elevation;
}