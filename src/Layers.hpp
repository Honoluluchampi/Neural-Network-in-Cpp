#pragma once
#include <vector>


class Layer
{
public:
    Layer(int size);
    ~Layer();

    // all layers should override these function
    virtual void Forward(std::vector<float> &input){};
    // for softmax
    virtual void Forward(std::vector<float> &input, std::vector<int> t){};
    virtual void Backward(std::vector<float> &input, float lr){};

protected:
    int mSize;
};


class Relu : public Layer
{
public:
    Relu(int size);

    void Forward(std::vector<float> &input)override;
    void Backward(std::vector<float> &input, float lr)override;

private:
    std::vector<int> mMask;
};


class Affine : public Layer
{
public:
    Affine(int size, int nextSize);

    void Forward(std::vector<float> &input)override;
    void Backward(std::vector<float> &input, float lr)override;
    void InitWeights(int nextSize);

private:
    std::vector<float> mInput;
    std::vector<std::vector<float>> mWeights;
    std::vector<float> mBias;
    std::vector<std::vector<float>> mDefWeights;
    std::vector<float> mDefBias;
};


class Softmax : public Layer
{
public:
    Softmax(int size);
    float LossForward(std::vector<float> &input, int t);
    std::vector<float> LossBackward();

private:
    float mLoss;
    std::vector<float> mOutput;
    int mLabel;
};