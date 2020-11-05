#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader.h>
#include <learnopengl/camera.h>

#include <iostream>
#include <vector>
#include <set>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vec3f.h"

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

__device__ __host__
inline float fmax3(float a, float b, float c)
{
    float t = a;
    if (b > t) t = b;
    if (c > t) t = c;
    return t;
}

__device__ __host__
inline float fmin3(float a, float b, float c)
{
    float t = a;
    if (b < t) t = b;
    if (c < t) t = c;
    return t;
}

__device__ __host__
inline int project3(vec3f& ax,
                    vec3f& p1, vec3f& p2, vec3f& p3)
{
    float P1 = ax.dot(p1);
    float P2 = ax.dot(p2);
    float P3 = ax.dot(p3);

    float mx1 = fmax3(P1, P2, P3);
    float mn1 = fmin3(P1, P2, P3);

    if (mn1 > 0) return 0;
    if (0 > mx1) return 0;
    return 1;
}

__device__ __host__
inline int project6(vec3f& ax,
                    vec3f& p1, vec3f& p2, vec3f& p3,
                    vec3f& q1, vec3f& q2, vec3f& q3)
{
    float P1 = ax.dot(p1);
    float P2 = ax.dot(p2);
    float P3 = ax.dot(p3);
    float Q1 = ax.dot(q1);
    float Q2 = ax.dot(q2);
    float Q3 = ax.dot(q3);

    float mx1 = fmax3(P1, P2, P3);
    float mn1 = fmin3(P1, P2, P3);
    float mx2 = fmax3(Q1, Q2, Q3);
    float mn2 = fmin3(Q1, Q2, Q3);

    if (mn1 > mx2) return 0;
    if (mn2 > mx1) return 0;
    return 1;
}

__device__ __host__
bool collide(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
    vec3f p1;
    vec3f p2 = P2 - P1;
    vec3f p3 = P3 - P1;
    vec3f q1 = Q1 - P1;
    vec3f q2 = Q2 - P1;
    vec3f q3 = Q3 - P1;

    vec3f e1 = p2 - p1;
    vec3f e2 = p3 - p2;
    vec3f e3 = p1 - p3;

    vec3f f1 = q2 - q1;
    vec3f f2 = q3 - q2;
    vec3f f3 = q1 - q3;

    vec3f n1 = e1.cross(e2);
    vec3f m1 = f1.cross(f2);

    vec3f g1 = e1.cross(n1);
    vec3f g2 = e2.cross(n1);
    vec3f g3 = e3.cross(n1);

    vec3f h1 = f1.cross(m1);
    vec3f h2 = f2.cross(m1);
    vec3f h3 = f3.cross(m1);

    vec3f ef11 = e1.cross(f1);
    vec3f ef12 = e1.cross(f2);
    vec3f ef13 = e1.cross(f3);
    vec3f ef21 = e2.cross(f1);
    vec3f ef22 = e2.cross(f2);
    vec3f ef23 = e2.cross(f3);
    vec3f ef31 = e3.cross(f1);
    vec3f ef32 = e3.cross(f2);
    vec3f ef33 = e3.cross(f3);

    vec3f p1_q1 = p1 - q1;
    vec3f p2_q1 = p2 - q1;
    vec3f p3_q1 = p3 - q1;
    if (!project3(n1, q1, q2, q3)) return false;
    if (!project3(m1, p1_q1, p2_q1, p3_q1)) return false;

    if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

    return true;
}

__global__ void kernel(const float* vertices,
                       const unsigned int* indices,
                       const unsigned int num_triangles,
                       unsigned int* flags)
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
bool displayCollision = false;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{
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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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
    Shader shader("E:/workspace/OpenGL/OpenGL/src/shaders/model.vs",
                  "E:/workspace/OpenGL/OpenGL/src/shaders/model.fs");

    // read triangles from obj file
    // ----------------------------
    const char* objFilePath = "E:/workspace/OpenGL/OpenGL/resources/two_spheres.obj";
    // const char* objFilePath = "E:/GPU-cuda-course/flag-no-cd/0181_00.obj";
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    if (!readObjFile(objFilePath, vertices, indices))
    {
        std::cout << "Failed to read obj file" << std::endl;
        glfwTerminate();
        return -1;
    }

    // collision detection on GPU
    // --------------------------
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

    //const dim3 BLOCK_SIZE(256, 256);
    //const dim3 GRID_SIZE((numTriangles + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
    //                     (numTriangles + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
    const dim3 BLOCK_SIZE(4, 4);
    const dim3 GRID_SIZE(2, 2);
    kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_vertices, d_indices, numTriangles, d_flags);

    std::vector<unsigned int> flags(numTriangles);
    cudaMemcpy(&flags[0], d_flags, sizeof(unsigned int) * numTriangles, cudaMemcpyDeviceToHost);

    std::vector<unsigned int> indices2;
    for (unsigned int i = 0; i < numTriangles; i++)
    {
        if (flags[i] > 0)
        {
            indices2.push_back(indices[i * 3]);
            indices2.push_back(indices[i * 3 + 1]);
            indices2.push_back(indices[i * 3 + 2]);
        }
    }

    cudaFree(d_vertices);
    cudaFree(d_indices);
    cudaFree(d_flags);

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

    unsigned int VBO, VAO, EBO, EBO2;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &EBO2);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)* vertices.size(), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)* indices2.size(), &indices2[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

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

        if (displayCollision)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
            glDrawElements(GL_TRIANGLES, indices2.size(), GL_UNSIGNED_INT, 0);
        }
        else
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
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

    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
        displayCollision = false;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
        displayCollision = true;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
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
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

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
