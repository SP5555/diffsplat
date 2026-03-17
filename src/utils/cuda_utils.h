#pragma once
#include <cstdio>
#include "ansi_colors.h"

#ifdef DEBUG

#define CUDA_WARN(err)                                                            \
    do                                                                            \
    {                                                                             \
        cudaError_t e = (err);                                                    \
        if (e != cudaSuccess)                                                     \
        {                                                                         \
            fprintf(stderr, ANSI_YELLOW "[CUDA] Error at %s:%d: %s\n" ANSI_RESET, \
                    __FILE__, __LINE__, cudaGetErrorString(e));                   \
        }                                                                         \
    } while (0)

#define CUDA_CHECK(err)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t e = (err);                                                 \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            fprintf(stderr, ANSI_RED "[CUDA] Error at %s:%d: %s\n" ANSI_RESET, \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_SYNC_CHECK()               \
    do                                  \
    {                                   \
        cudaDeviceSynchronize();        \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)

#else

#define CUDA_WARN(err) (err)

#define CUDA_CHECK(err) (err)

#define CUDA_SYNC_CHECK() \
    do                    \
    {                     \
    } while (0)

#endif // DEBUG

template <typename T>
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

    operator T *() { return ptr; }
    operator const T *() const { return ptr; }
};
