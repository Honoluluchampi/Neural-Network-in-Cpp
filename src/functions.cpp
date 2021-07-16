#include "functions.hpp"
#include <vector>
#include <iostream>
#include <string>


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
    for(int j = 0; j < matB.size(); j++)
    {
        for(int i = 0; i < vecA.size(); i++)
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
    for(int i = 0; i < vecA.size(); i++)
    {
        for(int j = 0; j < vecB.size(); j++)
        {
            output[i][j] = vecA[i] * vecB[j];
        }
    }
    return output;
}

std::vector<std::vector<float>> Transpose(std::vector<std::vector<float>> mat)
{
    std::vector<std::vector<float>> output(mat[0].size(), std::vector<float>(mat.size()));
    for (int i = 0; i < mat.size(); i++)
    {
        for(int j = 0; j < mat[0].size(); j++)
        {
            output[j][i] = mat[j][i];
        }
    }
    
    return output;
}

template<typename T>
std::vector<T> VecVecAdd(std::vector<T> vecA, std::vector<T> vecB)
{
    try
    {
        if(vecA.size() != vecB.size()) throw "size of vectors are different";
    }
    catch(std::string str)
    {
        std::cerr << str << std::endl;
    }
    
    std::vector<T> output(vecA.size());
    for (int i = 0; i < vecA.size(); i++)
    {
        output[i] = vecA[i] + vecB[i];
    }
    return output;
}