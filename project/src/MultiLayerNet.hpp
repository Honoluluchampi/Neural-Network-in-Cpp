#pragma once
#include <vector>
#include <memory>

class MultiLayerNet
{
public:
    MultiLayerNet(std::vector<int> layerSizeList);
    ~MultiLayerNet();

    // we should dont rewrite row-data, so copy value
    void SetLearningRate(float lr){mLearningRate = lr;}
    std::vector<float> Prediction(std::vector<float> input);
    float CalculateLoss(std::vector<float> predict, int label);
    float CalculateAccuracy(std::vector<float> predict, int label);
    void BackPropagation();

private:
    std::vector<int> mLayerSizeList;
    int mHiddenLayerNum;
    std::vector<class Layer*> mLayerList;
    class Softmax* mOutputLayer;
    float mLearningRate;
};