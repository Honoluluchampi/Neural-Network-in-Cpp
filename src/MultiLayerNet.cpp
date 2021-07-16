#include "MultiLayerNet.hpp"
#include <vector>

MultiLayerNet::MultiLayerNet(std::vector<int> layerSizeList):mLayerSizeList(layerSizeList)
{
    mHiddenLayerSize = mLayerSizeList.size()-2;
}