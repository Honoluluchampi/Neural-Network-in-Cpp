#include <vector>

class MultiLayerNet
{
public:
    MultiLayerNet(std::vector<int> layerSizeList);
    ~MultiLayerNet();

    // initilaze weights of each layers of each edge
    void InitializeWeights();
    std::vector<float>  Prediction();
    float CalculateLoss();
    float CalculateAcuracy();
    std::vector<float> BackPropagation();

private:
    std::vector<int> mLayerSizeList;
    int mHiddenLayerSize;
};