#include "linear.h"

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

LinearLayer::LinearLayer(int input_dim_, int output_dim_, std::string layer_name_)
{
    input_dim = input_dim_;
    output_dim = output_dim_;
    layer_name = layer_name_;
    weight = new float[input_dim*output_dim];
    bias = new float[1*output_dim];
};

LinearLayer::~LinearLayer()
{
    delete[] weight;
    delete[] bias;
};

Tensor LinearLayer::forward(Tensor input_tensor)
{
    std::vector<int> size{2, 4};
    Tensor output_tensor = Tensor(size);
    return output_tensor;
};

Tensor LinearLayer::backward(Tensor prev_d)
{
    std::vector<int> size{2, 4};
    Tensor output_tensor = Tensor(size);
    return output_tensor;
};

void LinearLayer::update_params()
{

};