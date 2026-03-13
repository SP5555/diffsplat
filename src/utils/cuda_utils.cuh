#pragma once
#include <cstdio>

#define CUDA_CHECK(err)                                         \
    do                                                          \
    {                                                           \
        cudaError_t e = (err);                                  \
        if (e != cudaSuccess)                                   \
        {                                                       \
            fprintf(stderr, "[CUDA] Error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

template<typename T>
struct CudaBuffer
{
    T *ptr = nullptr;
    size_t size = 0;

    CudaBuffer() = default;

    explicit CudaBuffer(size_t size_) { allocate(size_); }

    ~CudaBuffer() { free(); }

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr(other.ptr), size(other.size)
    {
        other.ptr = nullptr;
        other.size = 0;
    }
    CudaBuffer &operator=(CudaBuffer &&other) noexcept
    {
        if (this != &other)
        {
            free();
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void allocate(size_t size_)
    {
        free();
        size = size_;
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    }

    void free()
    {
        if (ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            size = 0;
        }
    }

    void zero()
    {
        if (ptr)
        {
            CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(T)));
        }
    }

    operator T*()             { return ptr; }
    operator const T*() const { return ptr; }
};

/* ===== utilities ===== */

void launchClampF(float* data, float min_val, float max_val, int n);