#include "functions.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>


// check the size of the vecA and matB before use this function
std::vector<float> VecMatDot(std::vector<float> vecA, std::vector<std::vector<float>> matB)
{
    try
    {
        if (vecA.size() != matB.size()) throw "vector and matrix size is not propoer";
    }

    catch(std::string &str)
    {
        std::cerr << str << '\n';
    }
    
    std::vector<float> output(matB[0].size());
    for(int j = 0; j < static_cast<int>(matB.size()); j++)
    {
        for(int i = 0; i < static_cast<int>(vecA.size()); i++)
        {
            output[j] += vecA[i] * matB[i][j];
        }
    }
    
    return output;
}

// i-vector * j-vector -> i*j-matrix
std::vector<std::vector<float>> VecVecDot(std::vector<float> vecA, std::vector<float> vecB)
{
    std::vector<std::vector<float>> output(vecA.size(), std::vector<float>(vecB.size()));
    for(int i = 0; i < static_cast<int>(vecA.size()); i++)
    {
        for(int j = 0; j < static_cast<int>(vecB.size()); j++)
        {
            output[i][j] = vecA[i] * vecB[j];
        }
    }
    return output;
}

std::vector<std::vector<float>> Transpose(std::vector<std::vector<float>> mat)
{
    std::vector<std::vector<float>> output(mat[0].size(), std::vector<float>(mat.size()));
    for (int i = 0; i < static_cast<int>(mat.size()); i++)
    {
        for(int j = 0; j < static_cast<int>(mat[0].size()); j++)
        {
            output[j][i] = mat[j][i];
        }
    }
    
    return output;
}

std::vector<float> VecVecAdd(std::vector<float> vecA, std::vector<float> vecB)
{
    try
    {
        if(vecA.size() != vecB.size()) throw "size of vectors are different";
    }
    catch(std::string str)
    {
        std::cerr << str << std::endl;
    }
    
    std::vector<float> output(vecA.size());
    for (int i = 0; i < static_cast<int>(vecA.size()); i++)
    {
        output[i] = vecA[i] + vecB[i];
    }
    return output;
}


std::vector<float> SoftmaxActivation(std::vector<float> input)
{
    // avoid Overflow
    auto maxVal = input[0];
    for(int i = 0; i < static_cast<int>(input.size()); i++)
    {
        if(maxVal < input[i]) maxVal = input[i];
    }
    float expSum = 0.0f;
    for(int i = 0; i < static_cast<int>(input.size()); i++)
    {
        input[i] -= maxVal;
        input[i] = std::exp(input[i]);
        expSum += input[i];
    }

    for(int i = 0; i < static_cast<int>(input.size()); i++)
    {
        input[i] /= expSum;
    }

    return input;
}

float CrossEntropyError(std::vector<float> output, int label)
{
    try
    {
        if(label >= static_cast<int>(output.size()) || label < 0) throw "output and label size is not proper";
    }
    catch(std::string &str)
    {
        std::cerr << str << std::endl;
    }
    
    float ans = -std::log(output[label]);

    return ans;
}