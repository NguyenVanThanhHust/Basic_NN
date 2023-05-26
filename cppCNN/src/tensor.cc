#include "tensor.h"

Tensor::Tensor(std::vector<int> size_)
{
    size = size_;
    int total_size=1;
    for (size_t i = 0; i < size.size(); i++)
    {
        total_size = total_size * size[i];
    }
    data = new float[total_size];
};

Tensor::~Tensor()
{
    delete[] data;
};
