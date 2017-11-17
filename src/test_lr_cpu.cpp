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
  MatrixXd data_test;
  VectorXd labels_test,predicted_labels;
  bool initialized=false;
  int test_rows = utils.get_Rows("MNIST_DATA/MNIST_test_labels.csv");
  int test_cols = utils.get_Cols("MNIST_DATA/MNIST_test_values.csv", ',');
     
  utils.read_Data("MNIST_DATA/MNIST_test_values.csv",data_test,test_rows,test_cols);
  utils.read_Labels("MNIST_DATA/MNIST_test_labels.csv",labels_test,test_rows);
      
  CPU_LogisticRegression logistic_regression;
  bool normalize=true;
  for(int epochs=0;epochs<10;epochs++){
    for(int num_batches=0;num_batches<45;num_batches++){
      string positive_data_extension = "_values_"+to_string(num_batches % 5);
      string positive_label_extension = "_labels_"+to_string(num_batches % 5);
      string negative_data_extension = "_values_"+to_string(num_batches);
      string negative_label_extension = "_labels_"+to_string(num_batches);
      string positive_data_name = "MNIST_DATA/train_positive";
      string negative_data_name = "MNIST_DATA/train_negative";
      MatrixXd positive_data;
      VectorXd positive_labels;
      int positive_rows = utils.get_Rows(positive_data_name+positive_label_extension);
      int positive_cols = utils.get_Cols(positive_data_name+positive_data_extension, ',');
      utils.read_Data(positive_data_name+positive_data_extension,positive_data,positive_rows,positive_cols);
      utils.read_Labels(positive_data_name+positive_label_extension,positive_labels,positive_rows);
      MatrixXd negative_data;
      VectorXd negative_labels;
      int negative_rows = utils.get_Rows(negative_data_name+negative_label_extension);
      int negative_cols = utils.get_Cols(negative_data_name+negative_data_extension, ',');
      utils.read_Data(negative_data_name+negative_data_extension,negative_data,negative_rows,negative_cols);
      utils.read_Labels(negative_data_name+negative_label_extension,negative_labels,negative_rows);
      MatrixXd data(0, positive_data.cols());;
      VectorXd labels(0);
      double ratio = (double)positive_rows/(double)negative_rows;
      data.resize(positive_data.rows()+negative_data.rows(), NoChange);
      labels.resize(positive_labels.rows()+negative_labels.rows());
      data << positive_data,negative_data;
      labels << positive_labels,negative_labels;
      cout << "----------------------------------------------" << endl;
      cout << "Read Mini Batch : "<<  to_string(num_batches) << endl;
      utils.dataPermutation(data, labels);
      cout << "Train" << endl;
      if(!initialized) {
        logistic_regression.init(data, labels, lambda, true, true, true);
        initialized=true;
      }
      else{
        logistic_regression.setData(data, labels);
      }
      logistic_regression.train(1e2,0.99);

     
      predicted_labels = logistic_regression.predict(data_test, false,normalize);
      normalize=false;
      utils.report(labels_test, predicted_labels, true);
      utils.calculateAccuracyPercent(labels_test, predicted_labels);
      utils.confusion_matrix(labels_test, predicted_labels, true);
    }
  }
  //utils.report(labels_test, predicted_labels, true);
  //utils.confusion_matrix(labels_test, predicted_labels, true);
  return 0;
}