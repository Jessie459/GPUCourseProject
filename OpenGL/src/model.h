#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>

class Model
{
public:
    Model(const std::vector<float>& vertices, const std::vector<unsigned int>& indices)
        : vertices(vertices), indices(indices)
    {
        initialize();
    }

    void setCPUCollision(const std::vector<unsigned int>& indices)
    {
        if (indices.size() == 0)
            return;
        cpuIndices = indices;

        glGenBuffers(1, &cpuEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cpuEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);
    }

    void setGPUCollision(const std::vector<unsigned int>& indices)
    {
        if (indices.size() == 0)
            return;
        gpuIndices = indices;

        glGenBuffers(1, &gpuEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);
    }

    void setBVHCollision(const std::vector<unsigned int>& indices)
    {
        if (indices.size() == 0)
            return;
        bvhIndices = indices;

        glGenBuffers(1, &bvhEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);
    }

    void draw()
    {
        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    }

    void drawCPUCollision()
    {
        if (cpuIndices.size() == 0)
            return;

        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cpuEBO);
        glDrawElements(GL_TRIANGLES, cpuIndices.size(), GL_UNSIGNED_INT, 0);
    }

    void drawGPUCollision()
    {
        if (gpuIndices.size() == 0)
            return;

        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuEBO);
        glDrawElements(GL_TRIANGLES, gpuIndices.size(), GL_UNSIGNED_INT, 0);
    }

    void drawBVHCollision()
    {
        if (bvhIndices.size() == 0)
            return;

        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhEBO);
        glDrawElements(GL_TRIANGLES, bvhIndices.size(), GL_UNSIGNED_INT, 0);
    }

private:
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> cpuIndices;
    std::vector<unsigned int> gpuIndices;
    std::vector<unsigned int> bvhIndices;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    unsigned int cpuEBO;
    unsigned int gpuEBO;
    unsigned int bvhEBO;

    void initialize()
    {
        // generate arrays and buffers
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        // bind VAO, VBO, EBO
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);

        // set vertex attribute pointers
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    }
};