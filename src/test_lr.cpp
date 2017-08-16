#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
//#include "likelihood/CPU_logistic_regression.hpp"
#include "likelihood/Mask_CPU_logistic_regression.hpp"
#include "utils/c_utils.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;


int main()
{
  C_utils utils;

  string data_csv_path, labels_csv_path;
  MatrixXd data;
  VectorXd labels;
  MatrixXd data_train, data_test;
  VectorXd labels_train, labels_test;

  data_csv_path = "sepal_x.csv";
  labels_csv_path = "sepal_y.csv";
  
  cout << "Read Data" << endl;

  utils.read_Data(data_csv_path,data_train);
  utils.read_Labels(labels_csv_path,labels_train);
  data_test = data_train;
  labels_test = labels_train;

  cout << "Train" << endl;
  double lambda = 100.0;
  Mask_CPU_LogisticRegression logistic_regression;
  logistic_regression.init(data_train, labels_train, lambda, false, true, true);
  logistic_regression.train(1000,0.001);
  /*VectorXd x(3);
  x << 10.,20.,30.;
  VectorXd temp = x.tail(x.rows()-1);
  logistic_regression.setWeights(temp);
  logistic_regression.setBias(x(0));
  logistic_regression.preCompute();
  VectorXd gradWeights = logistic_regression.computeGradient();
  double gradBias = logistic_regression.getGradientBias();
  VectorXd grad(3);
  grad << gradBias, gradWeights;
  cout << grad.transpose() << endl;*/
 
  cout << "Predict" << endl;
  VectorXd predicted_labels = logistic_regression.predict(data_test, false);
  utils.report(labels_test, predicted_labels, true);
  utils.calculateAccuracyPercent(labels_test, predicted_labels);
  utils.confusion_matrix(labels_test, predicted_labels, true);
  return 0;
}