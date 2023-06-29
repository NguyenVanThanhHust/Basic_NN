#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <random>
#include <string>
#include <vector>

class Tensor 
{
public:
    std::vector<int> size;
    float *data;
    Tensor(std::vector<int> size_);
    ~Tensor();
    float operator[](int postion);
};

#endif