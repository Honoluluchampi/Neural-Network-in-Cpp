#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

#include "MultiLayerNet.hpp"
#include "Layers.hpp"

std::vector<float> Line2Float(std::string &line)
{
  std::vector<float> values;
  std::string tmp = "";

  for(int i = 0; i < static_cast<int>(line.length()); i++)
  {
    if('0' <= int(line[i]) && int(line[i]) <= '9')
    {
      tmp += line[i];
    }
    else if(tmp.length() > 0)
    {
      values.push_back(stof(tmp));
      tmp = "";
    }
  }
  {
    if (tmp.length() > 0)
    {
      values.push_back(stof(tmp));
      tmp = "";
    }

    return values;
  }
}


// input arts : mnist_data_train, mnist_data_test, iter_num, learning_rate,
int main(int argc, char **argv)
{
  if(argc != 5) 
  {
    std::cout << "Error: command-line argument count mismatch. \n ./neuralnetwork <TRAIN_DATA> <TEST_DATA> <ITER_NUM> <LEARNING_RATE>" << std::endl;
    return 1;
  }

  std::string train_filename = argv[1];
  std::string test_filename = argv[2];
  std::ifstream train_infile(train_filename.c_str());
  std::ifstream test_infile(test_filename.c_str());
  std::vector<std::vector<float>> train_data;
  std::vector<int> train_labels;
  std::vector<std::vector<float>> test_data;
  std::vector<int> test_labels;
  std::string line;
  
  if(!train_infile.is_open())
  {
    std::cout << "Error: Failed to open file." << std::endl;
    return 1;
  }
  while(getline(train_infile, line))
  {
    std::vector<float> data = Line2Float(line);
    train_labels.push_back(data[0]);
    data.erase(data.begin());
    train_data.push_back(data);
  }

  train_infile.close();

  if(!test_infile.is_open())
  {
    std::cout << "Error: Failed to open file." << std::endl;
    return 1;
  }

  while(getline(test_infile, line))
  {
    std::vector<float> data = Line2Float(line);
    test_labels.push_back(data[0]);
    data.erase(data.begin());
    test_data.push_back(data);
  }

  test_infile.close();
  
  int iter_num = atoi(argv[3]);
  std::cout << iter_num << std::endl;
  float learning_rate = atof(argv[4]);
  std::cout << learning_rate << std::endl;

  int train_data_num = train_data.size();
  // log
  //std::vector<float> train_loss_list;
  //std::vector<float> train_acc_list;
  //std::vector<float> test_loss_list;
  //std::vector<float> test_acc_list;

  // learning
  std::vector<int> layer_size_list = {28*28, 10};
  MultiLayerNet* myNet = new MultiLayerNet(layer_size_list);
  myNet->SetLearningRate(learning_rate);

  for(int epoch = 0; epoch < iter_num; epoch++)
  {
    std::cout << "epoch" << std::endl;
    float accuracy = 0.0f;
    float loss = 0.0f;
    for(int i = 0; i < train_data_num; i++)
    {
      auto pred = myNet->Prediction(train_data[i]);
      std::cout << "pred" << std::endl;
      loss += myNet->CalculateLoss(pred, train_labels[i]);
      std::cout << "loss" << std::endl;
      accuracy += myNet->CalculateAccuracy(pred, train_labels[i]);
      std::cout << "acc" << std::endl;
      myNet->BackPropagation();
      std::cout << "back" << std::endl;
    }

    accuracy /= static_cast<float>(train_data_num);
    loss /= static_cast<float>(train_data_num);

    if(epoch % 10 == 0)
    {
      std::cout << "epoch : " << epoch << ". train_loss : " << loss << ". train_acc : " << accuracy << std::endl;
    }
  }
}
