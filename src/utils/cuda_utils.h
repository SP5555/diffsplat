#pragma once

#define CUDA_FREE(ptr)     \
    do                     \
    {                      \
        if (ptr)           \
        {                  \
            cudaFree(ptr); \
            ptr = nullptr; \
        }                  \
    } while (0)

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
