#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <cmath>

#include  <stdio.h>
#include <direct.h>

#include "vec3f.h"
#include "bvh.h"
#include "camera.h"
#include "shader.h"
#include "model.h"

bool readObjFile(const char* path, std::vector<float>& vertices, std::vector<unsigned int>& indices);
__global__ void kernel(const float* vertices, const unsigned int* indices, const unsigned int num_triangles, unsigned int* flags);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

bool displayBVHCollision = false;
bool checkedBVHCollision = false;
bool displayNaiveCollision = false;
bool checkedNaiveCollision = false;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "usage: " << argv[0] << " path" << std::endl;
        return 0;
    }
    const char* objFilePath = argv[1];

    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> naiveIndices;
    std::vector<unsigned int> bvhIndices;

    // read triangles from obj file
    // ----------------------------
    if (!readObjFile(objFilePath, vertices, indices))
    {
        std::cout << "Failed to read obj file" << std::endl;
        return -1;
    }
    std::cout << "triangle number = " << indices.size() / 3 << std::endl;

    // collision detection on GPU
    // --------------------------
    

    // collision detection on CPU
    // --------------------------
    /*std::set<unsigned int> s;
    unsigned int count = 0;

    #pragma omp parallel for
    for (int i = 0; i < numTriangles; i++)
    {
        unsigned int i0 = indices[i * 3];
        unsigned int i1 = indices[i * 3 + 1];
        unsigned int i2 = indices[i * 3 + 2];

        vec3f p0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        vec3f p1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        vec3f p2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        for (int j = 0; j < numTriangles; j++)
        {
            if (i >= j)
                continue;

            unsigned int j0 = indices[j * 3];
            unsigned int j1 = indices[j * 3 + 1];
            unsigned int j2 = indices[j * 3 + 2];

            if (i0 == j0 || i0 == j1 || i0 == j2)
                continue;
            if (i1 == j0 || i1 == j1 || i1 == j2)
                continue;
            if (i2 == j0 || i2 == j1 || i2 == j2)
                continue;

            vec3f q0(vertices[j0 * 3], vertices[j0 * 3 + 1], vertices[j0 * 3 + 2]);
            vec3f q1(vertices[j1 * 3], vertices[j1 * 3 + 1], vertices[j1 * 3 + 2]);
            vec3f q2(vertices[j2 * 3], vertices[j2 * 3 + 1], vertices[j2 * 3 + 2]);

            if (collide(p0, p1, p2, q0, q1, q2))
            {
                s.insert(i);
                s.insert(j);
                ++count;
            }
        }
    }
    std::cout << count << std::endl;

    std::vector<unsigned int> indices2(s.size() * 3);
    unsigned int idx = 0;
    for (auto iter = s.begin(); iter != s.end(); iter++)
    {
        indices2[idx * 3 + 0] = indices[(*iter) * 3 + 0];
        indices2[idx * 3 + 1] = indices[(*iter) * 3 + 1];
        indices2[idx * 3 + 2] = indices[(*iter) * 3 + 2];
        ++idx;
    }*/

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Viewer", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // build and compile shaders
    // -------------------------
    Shader shader("./src/shaders/model.vs", "./src/shaders/model.fs");

    // load cloth model
    // ----------------
    Model cloth(vertices, indices);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        shader.setMat4("projection", projection);
        shader.setMat4("view", view);
        shader.setMat4("model", model);

        if (displayNaiveCollision == false && displayBVHCollision == false)
        {
            cloth.draw();
        }
        else if (displayNaiveCollision == true)
        {
            if (checkedNaiveCollision == false)
            {
                unsigned int numTriangles = indices.size() / 3;

                float* d_vertices;
                unsigned int* d_indices;
                unsigned int* d_flags;

                cudaMalloc(&d_vertices, sizeof(float) * vertices.size());
                cudaMalloc(&d_indices, sizeof(unsigned int) * indices.size());
                cudaMalloc(&d_flags, sizeof(unsigned int) * numTriangles);

                cudaMemcpy(d_vertices, &vertices[0], sizeof(float) * vertices.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_indices, &indices[0], sizeof(unsigned int) * indices.size(), cudaMemcpyHostToDevice);
                cudaMemset(d_flags, 0, sizeof(unsigned int) * numTriangles);

                const dim3 BLOCK_SIZE(32, 32);
                const dim3 GRID_SIZE(64, 64);

                cudaEvent_t start, stop;
                float elapsedTime;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                std::cout << "\ncollision detection using naive method..." << std::endl;
                cudaEventRecord(start, 0);
                kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_vertices, d_indices, numTriangles, d_flags);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "elapsed time: " << elapsedTime / 1000.0f << " seconds" << std::endl;
                std::cout << "collision detection using naive method done" << std::endl;

                std::vector<unsigned int> flags(numTriangles);
                cudaMemcpy(&flags[0], d_flags, sizeof(unsigned int) * numTriangles, cudaMemcpyDeviceToHost);

                unsigned int counter = 0;
                for (unsigned int i = 0; i < numTriangles; i++)
                {
                    if (flags[i] > 0)
                    {
                        naiveIndices.push_back(indices[i * 3]);
                        naiveIndices.push_back(indices[i * 3 + 1]);
                        naiveIndices.push_back(indices[i * 3 + 2]);
                        ++counter;
                    }
                }
                cloth.setNaiveCollision(naiveIndices);
                std::cout << "collision triangle number = " << counter << std::endl;

                cudaFree(d_vertices);
                cudaFree(d_indices);
                cudaFree(d_flags);

                checkedNaiveCollision = true;
            }

            cloth.drawNaiveCollision();
        }
        else if (displayBVHCollision == true)
        {
            if (checkedBVHCollision == false)
            {
                cudaEvent_t start, stop;
                float elapsedTime;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                BVH tree(vertices, indices);

                std::cout << "\nconstructing bvh tree..." << std::endl;
                cudaEventRecord(start, 0);
                tree.constructBVH();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "elapsed time: " << elapsedTime / 1000.0f << " seconds" << std::endl;
                std::cout << "constructing bvh tree done" << std::endl;

                std::cout << "traversing bvh tree..." << std::endl;
                cudaEventRecord(start, 0);
                tree.traverseBVH();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                std::cout << "elapsed time: " << elapsedTime / 1000.0f << " seconds" << std::endl;
                std::cout << "traversing bvh tree done" << std::endl;

                std::vector<unsigned int> flags = tree.getFlags();

                unsigned int counter = 0;
                for (unsigned int i = 0; i < indices.size() / 3; i++)
                {
                    if (flags[i] > 0)
                    {
                        bvhIndices.push_back(indices[i * 3]);
                        bvhIndices.push_back(indices[i * 3 + 1]);
                        bvhIndices.push_back(indices[i * 3 + 2]);
                        ++counter;
                    }
                }
                cloth.setBVHCollision(bvhIndices);
                std::cout << "collision triangle number = " << counter << std::endl;

                checkedBVHCollision = true;
            }

            cloth.drawBVHCollision();
        }

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        displayNaiveCollision = false;
        displayBVHCollision = false;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        displayNaiveCollision = true;
        displayBVHCollision = false;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        displayNaiveCollision = false;
        displayBVHCollision = true;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

bool readObjFile(const char* path, std::vector<float>& vertices, std::vector<unsigned int>& indices)
{
    FILE* stream;
    errno_t err = fopen_s(&stream, path, "r");
    if (err)
        return false;

    char buffer[1024];
    while (fgets(buffer, 1024, stream))
    {
        if (buffer[0] == 'v' && buffer[1] == ' ') // vertices
        {
            float x, y, z;
            sscanf_s(buffer + 2, "%f%f%f", &x, &y, &z);
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
        else if (buffer[0] == 'f' && buffer[1] == ' ') // faces
        {
            unsigned int idx0, idx1, idx2, idx3;
            bool quad = false;

            char* next = buffer;
            sscanf_s(next + 2, "%u", &idx0);
            next = strchr(next + 2, ' ');
            sscanf_s(next + 1, "%u", &idx1);
            next = strchr(next + 1, ' ');
            sscanf_s(next + 1, "%u", &idx2);
            next = strchr(next + 1, ' ');
            if (next != NULL && next[1] >= '0' && next[1] <= '9')
            {
                if (sscanf_s(next + 1, "%u", &idx3))
                    quad = true;
            }

            --idx0;
            --idx1;
            --idx2;
            indices.push_back(idx0);
            indices.push_back(idx1);
            indices.push_back(idx2);

            if (quad) {
                --idx3;
                indices.push_back(idx0);
                indices.push_back(idx2);
                indices.push_back(idx3);
            }
        }
    }
    fclose(stream);
    if (vertices.size() == 0 || indices.size() == 0)
        return false;
    return true;
}

__global__ void kernel(const float* vertices, const unsigned int* indices, const unsigned int num_triangles, unsigned int* flags)
{
    unsigned int i_offset = blockDim.x * gridDim.x;
    unsigned int j_offset = blockDim.y * gridDim.y;

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < num_triangles)
    {
        unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
        while (j < num_triangles)
        {
            if (i >= j)
            {
                j += j_offset;
                continue;
            }

            unsigned int i0 = indices[i * 3];
            unsigned int i1 = indices[i * 3 + 1];
            unsigned int i2 = indices[i * 3 + 2];

            unsigned int j0 = indices[j * 3];
            unsigned int j1 = indices[j * 3 + 1];
            unsigned int j2 = indices[j * 3 + 2];

            if (i0 == j0 || i0 == j1 || i0 == j2 ||
                i1 == j0 || i1 == j1 || i1 == j2 ||
                i2 == j0 || i2 == j1 || i2 == j2)
            {
                j += j_offset;
                continue;
            }

            vec3f p0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
            vec3f p1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
            vec3f p2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

            vec3f q0(vertices[j0 * 3], vertices[j0 * 3 + 1], vertices[j0 * 3 + 2]);
            vec3f q1(vertices[j1 * 3], vertices[j1 * 3 + 1], vertices[j1 * 3 + 2]);
            vec3f q2(vertices[j2 * 3], vertices[j2 * 3 + 1], vertices[j2 * 3 + 2]);

            if (collide(p0, p1, p2, q0, q1, q2))
            {
                atomicAdd(&flags[i], 1);
                atomicAdd(&flags[j], 1);
            }

            j += j_offset;
        }

        i += i_offset;
    }
}
