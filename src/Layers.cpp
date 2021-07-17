#include <vector>
#include <iostream>
#include <random>
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

void Relu::Backward(std::vector<float> &input, float lr)
{
    for(auto idx : mMask)
    {
        input[idx] = 0;
    }
    
    // clear mask idx
    mMask.clear();
}


// Affine layer
Affine::Affine(int size, int nextSize) : Layer(size)
{
    InitWeights(nextSize);
    mBias = std::vector<float>(size, 0.0f);
}

void Affine::Forward(std::vector<float> &input)
{
    // check whether the size of matrix and vector is proper
    if(input.size() != mWeights.size())
    {
        std::cout << "incorrect matrix/vector size" << std::endl;
        return ;
    }
    mInput = input;
    input = VecVecAdd(VecMatDot(input, mWeights), mBias);
}

void Affine::Backward(std::vector<float> &input, float lr)
{
    if(input.size() != mWeights[0].size())
    {
        std::cout << "incorrect matrix/vector size" << std::endl;
        return ;
    }
    input = VecMatDot(input, Transpose(mWeights));
    mDefWeights = VecVecDot(mInput, input);
    mDefBias = input;
    for(int i = 0; i < static_cast<int>(mWeights.size()); i++)
    {
        for (int j = 0; j < static_cast<int>(mWeights[0].size()); j++)
        {
            mWeights[i][j] -= lr * mDefWeights[i][j];
        }
    }
    for(int i = 0; i < static_cast<int>(mBias.size()); i++)
    {
        mBias[i] -= lr * mDefBias[i];
    }
}

void Affine::InitWeights(int nextSize)
{
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(0.0,1.0);
    for(int i = 0; i < mSize; i++)
    {
        for(int j = 0; j < nextSize; j++)
        {
            mWeights[i].push_back(dist(engine));
        }
    }
}


// Softmax Layer
Softmax::Softmax(int size) : Layer(size)
{

}

float Softmax::LossForward(std::vector<float> &input, int label)
{
    mLabel = label;
    mOutput = SoftmaxActivation(input);
    mLoss = CrossEntropyError(mOutput, mLabel);
    return mLoss;
}

std::vector<float> Softmax::LossBackward()
{
    mOutput[mLabel] -= 1;
    return mOutput;
}