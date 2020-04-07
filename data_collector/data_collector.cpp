#include <iostream>
#include <fstream>
#include <random>
#include <cstdlib>
#include <thread>
#include <mutex>
#include "mujoco.h"
#include "glfw3.h"

// Multithreading.
std::mutex g_mutex;
const int thread_num = 10;
const int trajectories_num = 13000;

// MuJoCo data structures
mjModel *m;   // MuJoCo model.
mjData *d[thread_num];  // MuJoCo data.
mjvScene scn;           // Abstract scene.
mjrContext con;         // Custom GPU context.

// Uniform distribution generator.
std::random_device rd;
std::mt19937 gen{rd()};

std::uniform_real_distribution<double> distribution_obstacle(-9.0, 9.0);
std::uniform_real_distribution<double> distribution_direction(-1.0, 1.0);

// Indexes.
int slider_obstacle_jnt_idx;

// Log File.
std::ofstream dataLogfile;

/**
 * Resets the simulating data.
 *
 * @param t the index of a thread
 */
void reset(int t) {
    mj_resetData(m, d[t]);

    d[t]->qpos[slider_obstacle_jnt_idx] = distribution_obstacle(gen);
}

/**
 * Initializes the MuJoCo environment.
 *
 * @param modelPath the path of the model for simulation
 */
void init_mujoco(const char *modelPath) {
    // Activates MuJoCo.
    const char *mjKeyPath = std::getenv("MUJOCO_KEY");
    std::cout << "MuJoCo path:  " << mjKeyPath << std::endl;
    mj_activate(mjKeyPath);

    m = mj_loadXML(modelPath, nullptr, nullptr, 0);
    if (!m) mju_error("Model cannot be loaded");

    // Make data for all threads.
    for (auto &n : d) {
        n = mj_makeData(m);
    }

}

/**
 * Ends the current MuJoCo environment decently.
 */
void destruct() {
    // Frees visualization storage.
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Frees the data on all threads and
    for (auto &n : d) {
        mj_deleteData(n);
    }

    // Deletes the model.
    mj_deleteModel(m);

    // Deactivates MuJoCo.
    mj_deactivate();

    // Terminates GLFW.
    glfwTerminate();
}

/**
 * Collects the data obtained from the simuation.
 *
 * @param t the index of a thread
 */
void collect_data(int t) {
    // The loop for all trajectories.
    for (int j = 0; j < trajectories_num; j++) {
        // Resets the data.
        reset(t);

        std::string trajectory;

        double direction = distribution_direction(gen);
        double speed;

        if (direction < 0) {
            speed = -1;
        } else {
            speed = 1;
        }

        // The loop for each trajectory.
        for (int i = 0; i < 150; i++) {
            mjtNum simstart = d[t]->time;

            while (d[t]->time - simstart < 0.5) {
                mj_step1(m, d[t]);
                d[t]->ctrl[0] = speed;
                mj_step2(m, d[t]);
            }

            if (d[t]->qpos[slider_obstacle_jnt_idx] < -10) {
                speed = 1;
            }

            if (d[t]->qpos[slider_obstacle_jnt_idx] > 10) {
                speed = -1;
            }

            trajectory += std::to_string(d[t]->qpos[slider_obstacle_jnt_idx]);

            if (i != 149)
                trajectory += ",";
        }

        // Writes the current trajectory to the log file.
        g_mutex.lock();

        dataLogfile << trajectory << std::endl;

        g_mutex.unlock();

        // Shows the current working progress.
        if (((j + 1) % (trajectories_num / 100)) == 0) {
            std::cout << "Task #" << t << " has collected " << (j + 1) * 100 / trajectories_num
                      << "% data..." << std::endl;
        }
    }
}


/**
 * The main function.
 *
 * @param argc N/A
 * @param argv N/A
 * @return EXIT_SUCCESS
 */
int main(int argc, const char **argv) {
    // Loads and compiles model.
    const char *modelPath = "../../sim_env/mujoco_model/maze.xml";
    init_mujoco(modelPath);

    // Initializes the indexes.
    slider_obstacle_jnt_idx = mj_name2id(m, mjOBJ_JOINT, "slider:obstacle");

    const char *outputPath = "/home/xinchaosong/raw_data_102.csv";
    dataLogfile.open(outputPath);

    // Sets up all threads.
    std::thread task[thread_num];

    // direction = 0;

    for (int t = 0; t < thread_num; t++) {
        task[t] = std::thread(collect_data, t);
        std::cout << "Task #" << t << " starts" << std::endl;
    }

    for (int t = 0; t < thread_num; t++) {
        task[t].join();
        std::cout << "Task #" << t << " finished." << std::endl;
    }

    // Ends the work.
    destruct();
    dataLogfile.close();

    return EXIT_SUCCESS;
}
