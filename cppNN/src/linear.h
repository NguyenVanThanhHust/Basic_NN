#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tensor.h"

class LinearLayer
{
    int input_dim=1, output_dim=1;
    std::string layer_name;
    float *weight, *bias;

public:
    LinearLayer(int input_dim_, int output_dim_, std::string layer_name_="linear");
    ~LinearLayer();
    Tensor forward(Tensor input_tensor);
    Tensor backward(Tensor d_output);
    void update_params();
};

#endif