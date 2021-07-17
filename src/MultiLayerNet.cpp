#include "MultiLayerNet.hpp"
#include "Layers.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

MultiLayerNet::MultiLayerNet(std::vector<int> layerSizeList):mLayerSizeList(layerSizeList)
{
    mHiddenLayerNum = mLayerSizeList.size()-2;
    for(int i = 0; i < mHiddenLayerNum + 1; i++)
    {
        Affine* affine = new Affine(mLayerSizeList[i], mLayerSizeList[i+1]);
        mLayerList.push_back(affine);
        Relu* relu = new Relu(mLayerSizeList[i+1]);
        mLayerList.push_back(relu);
        // last layer
        mOutputLayer = new Softmax(mLayerSizeList[mLayerSizeList.size()-1]);
    }
}

std::vector<float> MultiLayerNet::Prediction(std::vector<float> input)
{
    for (auto &layer : mLayerList)
    {
        layer->Forward(input);
    }
    
    return input;
}

float MultiLayerNet::CalculateLoss(std::vector<float> predict, int label)
{
    return mOutputLayer->LossForward(predict, label);
}

// 1.0f or 0.0f
float MultiLayerNet::CalculateAccuracy(std::vector<float> predict, int label)
{
    auto maxIter = std::max_element(predict.begin(), predict.end());
    int argmax = static_cast<int>(std::distance(predict.begin(), maxIter));
    return static_cast<float>(argmax == label);
}

void MultiLayerNet::BackPropagation()
{
    auto dout = mOutputLayer->LossBackward();

    for(int i = mLayerList.size() -1 ; i >= 0; i--)
    {
        mLayerList[i]->Backward(dout, mLearningRate);
    }
}