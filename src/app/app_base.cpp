#include <iostream>
#include <stdexcept>

#include "app_base.h"
#include "../utils/ansi_colors.h"
#include "../loaders/image_saver.h"

/* ===== ===== GL Boilerplate ===== ===== */

static const float QUAD[] = {
    // x,    y,    u,    v
    -1.f, -1.f,  0.f,  1.f,
     1.f, -1.f,  1.f,  1.f,
     1.f,  1.f,  1.f,  0.f,
    -1.f, -1.f,  0.f,  1.f,
     1.f,  1.f,  1.f,  0.f,
    -1.f,  1.f,  0.f,  0.f,
};

static const char *VS_SRC = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

static const char *FS_SRC = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = vec4(texture(uTex, vUV).rgb, 1.0);
}
)glsl";

void AppBase::glfwErrorCallback(int error, const char *description)
{
    std::cerr << "[GLFW] Error " << error << ": " << description << "\n";
}

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "[GL] Shader error: " << log << "\n";
    }
    return s;
}

/* ===== ===== Lifecycle ===== ===== */

AppBase::AppBase(int width, int height, const std::string &title, bool resizable)
    : width(width), height(height), title(title), resizable(resizable)
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, resizable ? GLFW_TRUE : GLFW_FALSE);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout << "[App] " << ANSI_MAGENTA << "OpenGL " << glGetString(GL_VERSION) << ANSI_RESET << "\n"
              << "[App] " << ANSI_MAGENTA << "GL Renderer: " << glGetString(GL_RENDERER) << ANSI_RESET << "\n"
              << "[App] " << ANSI_MAGENTA << "CUDA Device: " << deviceProp.name << ANSI_RESET << "\n";

    initGL();

    cudaGLInteropSupported = checkCudaGLInterop();
    if (cudaGLInteropSupported)
        initPBO();
    else
        h_pixels.resize(width * height * 3);
    // something might have set an error here
    // that causes the whole debug build to crash
    // clear it
    cudaGetLastError();

    glfwSetWindowUserPointer(window, this);
    if (resizable) {
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow *win, int w, int h) {
            auto *app = static_cast<AppBase *>(glfwGetWindowUserPointer(win));
            if (!app) return;
            // update PBO/texture sizes
            app->onResize(w, h);
            // invoke the subclass's onWindowResize()
            // so it can react to the new size if needed
            app->onWindowResize(w, h);
        });
    }

    Input::install(window, &input);
}

AppBase::~AppBase()
{
    if (d_pbo_resource)
    {
        cudaGraphicsUnregisterResource(d_pbo_resource);
        d_pbo_resource = nullptr;
    
    }

    if (window)
        glfwDestroyWindow(window);
    glfwTerminate();
}

void AppBase::start()
{
    onStart();
    lastFrameTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        onRender();

        double currentTime  = glfwGetTime();
        double deltaTime    = currentTime - lastFrameTime;
        dt                  = static_cast<float>(deltaTime);
        lastFrameTime       = currentTime;
        timeSinceUpdate    += deltaTime;
        frameSinceUpdate++;

        if (timeSinceUpdate >= 0.1)
        {
            avgFPS = avgFPS * 0.4f + (frameSinceUpdate / (float)timeSinceUpdate) * 0.6f;
            char buf[128];
            snprintf(buf, sizeof(buf), "%s [FPS: %.1f]", title.c_str(), avgFPS);
            glfwSetWindowTitle(window, buf);
            timeSinceUpdate  = 0.0;
            frameSinceUpdate = 0;
        }

        glfwSwapBuffers(window);

        bool f12Pressed = (glfwGetKey(window, GLFW_KEY_F12) == GLFW_PRESS);
        if (f12Pressed && !f12WasPressed)
            saveScreenshot();
        f12WasPressed = f12Pressed;

        // exit on ESC
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        input.flush();
    }
}

/* ===== ===== Private Init ===== ===== */

void AppBase::initGL()
{
    shader_program  = GLShaderProgram{};
    texture         = GLTexture{};
    vao             = GLVertexArray{};
    vbo             = GLBuffer{};

    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // build shader program
    GLuint vs = compileShader(GL_VERTEX_SHADER,   VS_SRC);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, FS_SRC);
    glAttachShader(*shader_program, vs);
    glAttachShader(*shader_program, fs);
    glLinkProgram(*shader_program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void AppBase::initPBO()
{
    pbo = GLBuffer{};
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(float), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&d_pbo_resource, *pbo,
                                                    cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "[AppBase] PBO registration failed, falling back to host copy: "
                  << cudaGetErrorString(err) << "\n";
        pbo.reset();
        cudaGLInteropSupported = false;
        h_pixels.resize(width * height * 3);
    }
}

bool AppBase::checkCudaGLInterop()
{
    unsigned int deviceCount = 0;
    int glDevices[4];
    cudaGLGetDevices(&deviceCount, glDevices, 4, cudaGLDeviceListAll);

    int cudaDevice;
    cudaGetDevice(&cudaDevice);

    for (unsigned int i = 0; i < deviceCount; ++i)
        if (glDevices[i] == cudaDevice)
            return true;

    return false;
}

/* ===== ===== Resize ===== ===== */

void AppBase::onResize(int newWidth, int newHeight)
{
    if (newWidth <= 0 || newHeight <= 0)
        return;

    if (newWidth == width && newHeight == height)
        return;

    width  = newWidth;
    height = newHeight;

    // recreate PBO for new size
    if (pbo.has_value())
    {
        if (d_pbo_resource)
        {
            cudaGraphicsUnregisterResource(d_pbo_resource);
            d_pbo_resource = nullptr;
        }
        pbo.reset();
    }

    if (cudaGLInteropSupported)
        initPBO();
    else
        h_pixels.resize(newWidth * newHeight * 3);

    // resize texture
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, newWidth, newHeight, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glViewport(0, 0, newWidth, newHeight);
}

/* ===== ===== Display ===== ===== */

void AppBase::displayFrame(const float *d_pixels)
{
    lastPixels = d_pixels;

    if (cudaGLInteropSupported)
    {
        cudaGraphicsMapResources(1, &d_pbo_resource);
        float  *d_pbo    = nullptr;
        size_t  pbo_size = 0;
        cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &pbo_size, d_pbo_resource);
        cudaMemcpy(d_pbo, d_pixels, width * height * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &d_pbo_resource);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
        glBindTexture(GL_TEXTURE_2D, *texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        cudaMemcpy(h_pixels.data(), d_pixels,
                   width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, *texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, h_pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(*shader_program);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glBindVertexArray(*vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

/* ===== ===== Screenshot ===== ===== */
void AppBase::saveScreenshot()
{
    if (!lastPixels) return;
    time_t t = time(nullptr);
    char buf[64];
    strftime(buf, sizeof(buf), "screenshot_%y%m%d_%H%M%S.png", localtime(&t));
    saveScreenshot(buf);
}

void AppBase::saveScreenshot(const std::string &path)
{
    ImageSaver::saveAsPNG(lastPixels, width, height, path);
    std::cout << "[App] Screenshot saved: " << path << "\n";
}
