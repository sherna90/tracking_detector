#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "likelihood/CPU_logistic_regression.hpp"
#include "utils/c_utils.hpp"
#include <stdlib.h>

using namespace Eigen;
using namespace std;
using namespace cv;


int main(int argc, char *argv[] )
{
  double lambda;
  if ( argc != 2 ){
    lambda = 0.1;
  }
  else{
    lambda = atof(argv[1]);
  }
  C_utils utils;

  string data_csv_path, labels_csv_path,test_csv_path,labels_test_path;
  MatrixXd data;
  VectorXd labels;
  MatrixXd data_train, data_test;
  VectorXd labels_train, labels_test;

  data_csv_path = "MNIST_train_values.csv";
  test_csv_path = "MNIST_test_values.csv";
  labels_csv_path = "MNIST_train_labels.csv";
  labels_test_path = "MNIST_test_labels.csv";
  
  cout << "Read Data"  << endl;

  utils.read_Data(data_csv_path,data_train);
  utils.read_Data(test_csv_path,data_test);
  utils.read_Labels(labels_csv_path,labels_train);
  utils.read_Labels(labels_test_path,labels_test);

  cout << "Train" << endl;

  CPU_LogisticRegression logistic_regression;
  logistic_regression.init(data_train, labels_train, lambda, false, true, true);
  logistic_regression.train(1e4,0.9);

 
  cout << "Predict" << endl;
  VectorXd predicted_labels = logistic_regression.predict(data_test, false);
  int test_num=10;
  MatrixXd results=MatrixXd::Zero(test_num,2);
  results.col(0)=predicted_labels.head(test_num);
  results.col(1)=labels_test.head(test_num);
  MatrixXf::Index maxRow;
  //cout << results << endl;
  utils.report(labels_test, predicted_labels, true);
  utils.calculateAccuracyPercent(labels_test, predicted_labels);
  utils.confusion_matrix(labels_test, predicted_labels, true);
  return 0;
}