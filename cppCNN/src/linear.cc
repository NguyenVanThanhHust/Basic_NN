#include "linear.h"


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
    int batch_size = input_tensor.size[0];
    std::vector<int> output_size{batch_size, output_dim};
    Tensor output_tensor = Tensor(output_size);

    return output_tensor;
};

Tensor LinearLayer::backward(Tensor d_output)
{
    int batch_size = d_output.size[0];
};

void LinearLayer::update_params()
{

};