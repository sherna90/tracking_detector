#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
//#include "likelihood/CPU_logistic_regression.hpp"
#include "likelihood/CPU_logistic_regression.hpp"
#include "utils/c_utils.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;


int main()
{
  C_utils utils;

  string data_csv_path, labels_csv_path,test_csv_path,labels_test_path;
  MatrixXd data;
  VectorXd labels;
  MatrixXd data_train, data_test;
  VectorXd labels_train, labels_test;

  data_csv_path = "matlab_feature_train.csv";
  test_csv_path = "matlab_feature_test.csv";
  labels_csv_path = "matlab_label_train.csv";
  labels_test_path = "matlab_label_test.csv";
  
  cout << "Read Data" << endl;

  utils.read_Data(data_csv_path,data_train);
  utils.read_Data(test_csv_path,data_test);
  utils.read_Labels(labels_csv_path,labels_train);
  utils.read_Labels(labels_test_path,labels_test);

  cout << "Train" << endl;
  double lambda = 100.0;
  CPU_LogisticRegression logistic_regression;
  logistic_regression.init(data_train, labels_train, lambda, false, true, true);
  logistic_regression.train(1e3,0.99);

 
  cout << "Predict" << endl;
  VectorXd predicted_labels = logistic_regression.predict(data_test, true);
  int test_num=10;
  MatrixXd results=MatrixXd::Zero(test_num,2);
  results.col(0)=predicted_labels.head(test_num);
  results.col(1)=labels_test.head(test_num);
  MatrixXf::Index maxRow;
  double max = predicted_labels.maxCoeff(&maxRow);
  cout << max << endl;
  //utils.report(labels_test, predicted_labels, true);
  //utils.calculateAccuracyPercent(labels_test, predicted_labels);
  //utils.confusion_matrix(labels_test, predicted_labels, true);
  return 0;
}