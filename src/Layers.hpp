#include <vector>


class Layer
{
public:
    Layer(int size);
    ~Layer();

    // all layers should override these function
    virtual void Forward(std::vector<float> &input) = 0;
    virtual void Backward(std::vector<float> &input) = 0;
protected:
    int mSize;
};


class Relu : public Layer
{
public:
    Relu(int size);

    void Forward(std::vector<float> &input)override;
    void Backward(std::vector<float> &input)override;

private:
    std::vector<int> mMask;
};


class Affine : public Layer
{
public:
    Affine(int size, std::vector<std::vector<float>> weights, std::vector<float> bias);

    void Forward(std::vector<float> &input)override;
    void Backward(std::vector<float> &input)override;

private:
    std::vector<std::vector<float>> mWeights;
    std::vector<float> mBias;
    std::vector<std::vector<float>> mDefWeights;
    std::vector<float> mDefBias;
};