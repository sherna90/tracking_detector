#include "test_detector_cpu.hpp"

#ifndef PARAMS

const double GROUP_THRESHOLD = 0.0;
const double HIT_THRESHOLD = 0.99;
const double POSITIVE = 1.0;
const double NEGATIVE = 0.0;

#endif

TestDetector::TestDetector(){
	this->detector.init(GROUP_THRESHOLD, HIT_THRESHOLD);
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	this->generator.seed(seed1);
}

void TestDetector::generateFeatures(string train_path, string positive_list, string negative_list, string filename, double margin){
	string line;
	ifstream train_list((train_path+positive_list).c_str());
	if (!train_list) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
	cout << "positive" << endl;
	bool append = false;
	while (getline(train_list, line)) {
		string img_path = train_path+line;
		Mat current_frame = imread(img_path);
		Rect centerROI(margin, margin, current_frame.cols - margin*2, current_frame.rows - margin*2);
		Mat croppedImage = current_frame(centerROI);
		this->detector.generateFeatures(croppedImage, POSITIVE);
		this->detector.saveToCSV(filename+"positive_MARS", append);
		this->detector.dataClean();
		append = true;
  	}
  	append = false;
	cout << "negative" << endl;
	ifstream test_list((train_path+negative_list).c_str());
	if (!test_list) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
  	while (getline(test_list, line)) {
		string img_path = train_path+line;
		Mat current_frame = imread(img_path);
		cout << img_path << endl;
		this->detector.generateFeatures(current_frame, NEGATIVE);
		this->detector.saveToCSV(filename+"negative_MARS", append);
		this->detector.dataClean();
		append = true;
  	}
  	cout << "finish" << endl;
}

void TestDetector::train(){
	C_utils utils;
	string data_extension = "_values_"+to_string(183);
	string label_extension = "_labels_"+to_string(183);
	string positive_data_name = "MARS_DATA/train_positive";
	string negative_data_name = "MARS_DATA/train_negative";
	MatrixXd positive_data;
	VectorXd positive_labels;
	MatrixXd negative_data;
	VectorXd negative_labels;
	utils.read_Data(positive_data_name+data_extension,positive_data);
	utils.read_Labels(positive_data_name+label_extension,positive_labels);
	utils.read_Data(negative_data_name+data_extension,negative_data);
	utils.read_Labels(negative_data_name+label_extension,negative_labels);
	MatrixXd data_test(0, positive_data.cols());;
	VectorXd labels_test(0);
	data_test.resize(positive_data.rows()+negative_data.rows(), NoChange);
	labels_test.resize(positive_labels.rows()+negative_labels.rows());
	data_test << positive_data,negative_data;
	labels_test << positive_labels,negative_labels;
	bool data_processing=true;
	for(int epochs=0;epochs<1;epochs++){
		for(int num_batches=0;num_batches<183;num_batches++){
			data_extension = "_values_"+to_string(num_batches);
			label_extension = "_labels_"+to_string(num_batches);
			utils.read_Data(positive_data_name+data_extension,positive_data);
		 	utils.read_Labels(positive_data_name+label_extension,positive_labels);
		 	utils.read_Data(negative_data_name+data_extension,negative_data);
		 	utils.read_Labels(negative_data_name+label_extension,negative_labels);
			MatrixXd data(0, positive_data.cols());;
		 	VectorXd labels(0);
		 	data.resize(positive_data.rows()+negative_data.rows(), NoChange);
			labels.resize(positive_labels.rows()+negative_labels.rows());
			data << positive_data,negative_data;
			labels << positive_labels,negative_labels;
			cout << "----------------------------------------------" << endl;
			cout << "Read Mini Batch : "<<  to_string(num_batches) << endl;
			utils.dataPermutation(data, labels);
			cout << data.rows() << ", " << data.cols() << "," << labels.size() << endl;
			this->detector.loadFeatures(data, labels);
			this->detector.train();
			VectorXd predicted_labels=this->detector.predictTest(data_test,data_processing);
			data_processing=false;
			utils.report(labels_test, predicted_labels, true);
			utils.calculateAccuracyPercent(labels_test, predicted_labels);
			utils.confusion_matrix(labels_test, predicted_labels, true);
		}
		cout << "Epoch :"<< epochs << endl;
	}

}

void TestDetector::test_detector(string test_path, string positive_list, string negative_list){
	cout << "Positive" << endl;
	double positive_percent = this->detect(test_path, positive_list);
	cout << "Accuracy: "<< positive_percent << endl;
	cout << "Negative" << endl;	
	double negative_percent = this->detect(test_path, negative_list);
	cout << "Accuracy: "<< negative_percent << endl;
	
};


void TestDetector::loadModel(){
	C_utils utils;
	VectorXd mean;
	VectorXd std;
	VectorXd max;
	VectorXd min;
	VectorXd weights;
	VectorXd bias;
	utils.read_Labels("Model_means.csv",mean);
	utils.read_Labels("Model_weights.csv",weights);
	utils.read_Labels("Model_stds.csv",std);
	utils.read_Labels("Model_bias.csv",bias);
	utils.read_Labels("Model_maxs.csv",max);
	utils.read_Labels("Model_mins.csv",min);
	this->detector.loadModel(weights,mean,std, max, min, bias(0));
};
 
double TestDetector::detect(string train_path, string list){
	string line;
	ifstream detect_list((train_path+list).c_str());
	if (!detect_list) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
	vector<Rect> detections;
	namedWindow("Detector");
	while (getline(detect_list, line)) {
		string img_path = train_path+line;
		Mat current_frame = imread(img_path);
		Mat grayImg;
		detections.clear();
		//int newHeight = 200;
    	//int newWidth = current_frame.cols*newHeight/current_frame.rows;
		//resize(current_frame, current_frame, Size(newWidth, newHeight));
	    detections = this->detector.detect(current_frame);
		for (int i = 0; i < detections.size(); ++i){
				rectangle( current_frame, detections.at(i), Scalar(0,0,255), 2, LINE_8  );
		}
		imshow("Detector", current_frame);
		waitKey(0);

  	}
  	return 0;
};



int main(int argc, char* argv[]){
	
	string test_path = string("Pedestrians-Test/");
	string train_path = string("MARS/");
	string positive_list = string("pos.lst");
	string negative_list = string("neg.lst");


	TestDetector tracker = TestDetector();
	//tracker.generateFeatures(train_path, positive_list, negative_list, "train_", 0);
	tracker.loadModel();
	//tracker.train();
	//tracker.test_detector(train_path, positive_list, negative_list);
	tracker.detect(test_path,positive_list);
	//tracker.test();
}