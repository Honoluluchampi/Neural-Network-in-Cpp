#pragma once
#include <vector>

// dot function for vector and 2-dimensional matrix
std::vector<float> VecMatDot(std::vector<float> vecA, std::vector<std::vector<float>> matB);

// dot function for vector and vector
std::vector<std::vector<float>> VecVecDot(std::vector<float> vecA, std::vector<float> vecB);

// parallel dot function for 2-dimensional matrix
std::vector<float> ParallelAffineDot(std::vector<float> vecA, std::vector<std::vector<float>> matB);

// we dont like to change original matrix, so i dont use &
std::vector<std::vector<float>> Transpose(std::vector<std::vector<float>> mat);

std::vector<float> VecVecAdd(std::vector<float> vecA, std::vector<float> vecB);


// Softmax activation
std::vector<float> SoftmaxActivation(std::vector<float> input);

float CrossEntropyError(std::vector<float> output, int label);