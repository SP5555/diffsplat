#pragma once

#include <utility>
#include <glad/glad.h>

/* ===== ===== GL RAII Wrappers ===== ===== */

template<typename Derived>
struct GLObject
{
    GLuint id = 0;

    GLObject() = default;
    ~GLObject() { if (id) static_cast<Derived*>(this)->release(); }

    GLObject(const GLObject &)            = delete;
    GLObject &operator=(const GLObject &) = delete;

    GLObject(GLObject &&other) noexcept : id(other.id) { other.id = 0; }
    GLObject &operator=(GLObject &&other) noexcept
    {
        if (this != &other)
        {
            if (id) static_cast<Derived*>(this)->release();
            id = other.id;
            other.id = 0;
        }
        return *this;
    }

    operator GLuint() const { return id; }
};

struct GLBuffer : GLObject<GLBuffer>
{
    GLBuffer()  { glGenBuffers(1, &id); }
    void release() { glDeleteBuffers(1, &id); }
};

struct GLTexture : GLObject<GLTexture>
{
    GLTexture()  { glGenTextures(1, &id); }
    void release() { glDeleteTextures(1, &id); }
};

struct GLVertexArray : GLObject<GLVertexArray>
{
    GLVertexArray()  { glGenVertexArrays(1, &id); }
    void release() { glDeleteVertexArrays(1, &id); }
};

struct GLShaderProgram : GLObject<GLShaderProgram>
{
    GLShaderProgram()  { id = glCreateProgram(); }
    void release() { glDeleteProgram(id); }
};