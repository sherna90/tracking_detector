#include "test_detector_cpu.hpp"

#ifndef PARAMS

const double GROUP_THRESHOLD = 0.1;
const double HIT_THRESHOLD = 0.9;
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
		Rect centerROI(margin, margin, current_frame.cols - 2*margin, current_frame.rows - 2*margin);
		Mat croppedImage = current_frame(centerROI);
		this->detector.generateFeatures(croppedImage, POSITIVE);
		this->detector.saveToCSV(filename+"positive_INRIA", append);
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
		this->detector.saveToCSV(filename+"negative_INRIA", append);
		append = true;
  	}
  	cout << "finish" << endl;
}

void TestDetector::train(){
	C_utils utils;

	cout << "Read Data" << endl;
	string data_extension = "_values.csv";
	string label_extension = "_labels.csv";

	string positive_data_name = "train_positive_INRIA";
	string negative_data_name = "train_negative_INRIA";

	MatrixXd positive_data;
  	VectorXd positive_labels;

  	int positive_rows = utils.get_Rows(positive_data_name+label_extension);
  	int positive_cols = utils.get_Cols(positive_data_name+data_extension, ',');
  	
  	utils.read_Data(positive_data_name+data_extension,positive_data,positive_rows,positive_cols);
 	utils.read_Labels(positive_data_name+label_extension,positive_labels,positive_rows);
	
 	MatrixXd negative_data;
  	VectorXd negative_labels;

  	int negative_rows = utils.get_Rows(negative_data_name+label_extension);
  	int negative_cols = utils.get_Cols(negative_data_name+data_extension, ',');
 	
  	utils.read_Data(negative_data_name+data_extension,negative_data,negative_rows,negative_cols);
 	utils.read_Labels(negative_data_name+label_extension,negative_labels,negative_rows);
	
	MatrixXd data(0, positive_data.cols());;
 	VectorXd labels;
 	double ratio = (double)positive_rows/(double)negative_rows;
 	cout << "positive/negative ratio : " << ratio << endl;
 	//ratio=1.0;
 	if(ratio<0.5){
	 	data = positive_data;
	 	labels = positive_labels;
	 	double accept = 1.0 - ratio;
	 	uniform_real_distribution<double> unif(0.0,1.0);
	 	cout << "Data Rolling and Permutation" << endl;
	 	for (int i = 0; i < negative_rows; ++i)
	 	{
		 	double uni_rand = unif(this->generator);
			if(uni_rand>accept){ 
				data.conservativeResize(data.rows() + 1, NoChange);
				data.row(data.rows() - 1)=negative_data.row(i);
				labels.conservativeResize(labels.size() + 1 );
				labels(labels.size() - 1) = negative_labels(i);
			}
	 	}
	}
	else{
		data.resize(positive_data.rows()+negative_data.rows(), NoChange);
		labels.resize(positive_labels.rows()+negative_labels.rows());
		data << positive_data,negative_data;
		labels << positive_labels,negative_labels;
	}

 	utils.dataPermutation(data, labels);

 	//this->detector.loadFeatures(data, labels);
 	//cout << "init train" << endl;
	//this->detector.train();
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
	
	string test_path = string("INRIA/");
	string train_path = string("INRIA/");
	string positive_list = string("pos.lst");
	string negative_list = string("neg.lst");


	TestDetector tracker = TestDetector();
	tracker.generateFeatures(train_path, positive_list, negative_list, "train_", 3);
	//tracker.train();
	//tracker.loadModel();
	//tracker.test_detector(train_path, positive_list, negative_list);
	//tracker.detect(test_path,positive_list);
	//tracker.test();
}