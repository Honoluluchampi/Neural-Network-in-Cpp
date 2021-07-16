#include <vector>
#include <iostream>
#include "Layers.hpp"
#include "functions.hpp"

// General Layer Class
Layer::Layer(int size) : mSize(size)
{

}


// Relu Class
Relu::Relu(int size) : Layer(size)
{

}

void Relu::Forward(std::vector<float> &input)
{
    // mainus value should be zero.
    for(int i; i < mSize; i++)
    {
        if(input[i] < 0) mMask.push_back(i);
    }

    for(auto idx : mMask)
    {
        input[idx] = 0;
    }
}

void Relu::Backward(std::vector<float> &input)
{
    for(auto idx : mMask)
    {
        input[idx] = 0;
    }
    
    // clear mask idx
    mMask.clear();
}


// Affine layer
Affine::Affine(int size, std::vector<std::vector<float>> weights, std::vector<float> bias) : Layer(size)
{
    mWeights = weights;
    mBias = bias;
}

void Affine::Forward(std::vector<float> &input)
{
    // check whether the size of matrix and vector is proper
    if(input.size() != mWeights.size())
    {
        std::cout << "incorrect matrix/vector size" << std::endl;
        return ;
    }
    input = VecVecAdd<float>(VecMatDot(input, mWeights), mBias);
}

void Affine::Backward(std::vector<float> &input)
{
    if(input.size() != mWeights[0].size())
    {
        std::cout << "incorrect matrix/vector size" << std::endl;
        return ;
    }
    input = VecMatDot(input, Transpose(mWeights));
    mDefWeights = VecMatDot()
}