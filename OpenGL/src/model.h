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

    void setCollision(const std::vector<unsigned int>& indices)
    {
        if (indices.size() == 0)
            return;
        collideIndices = indices;

        glGenBuffers(1, &collideEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, collideEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * collideIndices.size(), &collideIndices[0], GL_STATIC_DRAW);
    }

    void draw()
    {
        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    }

    void drawCollision()
    {
        if (collideIndices.size() == 0)
            return;

        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, collideEBO);
        glDrawElements(GL_TRIANGLES, collideIndices.size(), GL_UNSIGNED_INT, 0);
    }

private:
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> collideIndices;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    unsigned int collideEBO;

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