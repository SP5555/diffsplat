#pragma once

class Layer
{
public:
    virtual ~Layer() = default;
    virtual void forward()   = 0;
    virtual void backward()  = 0;
    virtual void zero_grad() = 0;
    virtual void free()      = 0;
};