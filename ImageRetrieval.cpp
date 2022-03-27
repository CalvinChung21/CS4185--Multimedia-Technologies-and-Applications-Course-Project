
//* CS4185/CS5185 Multimedia Technologies and Applications
//* Course Assignment
//* Image Retrieval Project
//*/

#include "bof.h"
#include <fstream>
#define IMAGE_LIST_FILE "inputimage.txt"  // NOTE: this is relative to current file



// * Code copied from opencv website, found at:
// * https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
// */
Mat makeHistogram(Mat img_input, int histSize[], const float* ranges[]) {
	Mat hist;
	calcHist(&img_input, 1, 0, Mat(), hist, 1, histSize, ranges, true, false);
	normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

	return hist;
}


// * Code copied from Stack Overflow website, found at:
// *	https://stackoverflow.com/questions/32952577/calculating-cumulative-histogram
// */
Mat accumulatedHist(Mat hist_src, int num_bins) {
	Mat accumulatedHist = hist_src.clone();
	for (int i = 1; i < num_bins; i++)
		accumulatedHist.at<float>(i) += accumulatedHist.at<float>(i - 1);
	return accumulatedHist;
}

int main(){
	//* the Bag of Features Framework is copied and slightly modified to suit the work on my project and found at:
	//* https://github.com/AstarLight/Bag-of-Features-Framework
	//*/
	//used to build the dictionary, clustering and training the svm
	//the left input is class number and the right input is the image number of that class
	//only need to be used when you need to train new data
	//BuildDictionary(10,100);  

	//initilization
	Mat src_input;
	Mat db_img;
	int number;
	string path = "";
	int db_id = 0;
	double score[1000];
	double maxscore = 0;
	int maxscore_num;
	char maximg_name[200];
	int label[1000];
	bool labelCheck[1000];

	// initialization for hsv histogram
	int h_bins = 180, s_bins = 256;
	int histSize_hue[] = { h_bins };
	int histSize_sat[] = { s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges_hue[] = { h_ranges };
	const float* ranges_sat[] = { s_ranges };

	// used to load the dictionary and classifier data to the SVM
    TrainingDataInit();
	// used to calculate the accuracy of the classifier
	// the accuracy is about 92% when using the original database data not including the src input images
	// 89% after including the src input images and some similar pictuces to the src input from online to train the SVM
	// only need to be used when you want to calculate the accuracy of the SVM prediction
    //TestClassify(); 

	printf("1: beach\n");
	printf("2: building\n");
	printf("3: bus\n");
	printf("4: dinosaur\n");
	printf("5: flower\n");
	printf("6: horse\n");
	printf("7: man\n");
	printf("Type in the number to choose a category and type enter to confirm\n");
	scanf_s("%d", &number);


	switch (number) {
	case 1:
		src_input = imread("beach.jpg");  // read input image
		printf("You choose: %d - beach\n", number);
		path = "./Similar_img_1/"; // the path to store similar images
		break;
	case 2:
		src_input = imread("building.jpg");
		printf("You choose: %d - building\n", number);
		path = "./Similar_img_2/";
		break;
	case 3:
		src_input = imread("bus.jpg");
		printf("You choose: %d - bus\n", number);
		path = "./Similar_img_3/";
		break;
	case 4:
		src_input = imread("dinosaur.jpg");
		printf("You choose: %d - dinosaur\n", number);
		path = "./Similar_img_4/";
		break;
	case 5:
		src_input = imread("flower.jpg");
		printf("You choose: %d - flower\n", number);
		path = "./Similar_img_5/";
		break;
	case 6:
		src_input = imread("horse.jpg");
		printf("You choose: %d - horse\n", number);
		path = "./Similar_img_6/";
		break;
	case 7:
		src_input = imread("man.jpg");
		printf("You choose: %d - man\n", number);
		path = "./Similar_img_7/";
		break;
	}

	if (!src_input.data)
	{
		printf("Cannot find the input image!\n");
		system("pause");
		return -1;
	}
	imshow("Input", src_input);

	// convert the source image to hsv color and split its color channels
	Mat hsv_input;
	cvtColor(src_input, hsv_input, COLOR_BGR2HSV);
	vector<Mat> hsv_planes;
	split(hsv_input, hsv_planes);
	// calculate the hue and saturation histograms for source image 
	Mat hist_src_hue = makeHistogram(hsv_planes[0], histSize_hue, ranges_hue);
	Mat hist_src_sat = makeHistogram(hsv_planes[1], histSize_sat, ranges_sat);
	// calculate the accumulated hue and saturation histograms for source image
	Mat accumulatedHist_src_hue = accumulatedHist(hist_src_hue, h_bins);
	Mat accumulatedHist_src_sat = accumulatedHist(hist_src_sat, s_bins);
	// classify the source image
	int src_label = invoice_classify(src_input);

	FILE* fp;
	ifstream labelFile("classifiedResult.txt");
	// check if the label data for all database images is already saved in a file
	// if not then create a new file storing the label data for all database images
	if (!labelFile.is_open()) {
		ofstream myfile("classifiedResult.txt");
		//Read Database
		char imagepath[200];
		fopen_s(&fp, IMAGE_LIST_FILE, "r");
		printf("Extracting features from input images...\n");
		while (!feof(fp))
		{
			while (fscanf_s(fp, "%s ", imagepath, sizeof(imagepath)) > 0)
			{
				printf("%s\n", imagepath);
				char tempname[200];
				sprintf_s(tempname, 200, "../%s", imagepath);

				db_img = imread(tempname); // read database image
				if (!db_img.data)
				{
					printf("Cannot find the database image number %d!\n", db_id);
					system("pause");
					return -1;
				}
				label[db_id] = invoice_classify(db_img);
				myfile << label[db_id] << endl;

				if (src_label == label[db_id]) {
					labelCheck[db_id] = true;
				}
				else {
					labelCheck[db_id] = false;
				}

				// save all database images with a same label as the source image to a file
				if (labelCheck[db_id]) {
					string img_name = to_string(db_id);
					imwrite(path + img_name + ".jpg", db_img);
				}
				// convert the database image to hsv color and split its color channels
				Mat db_hsv_img;
				cvtColor(db_img, db_hsv_img, COLOR_BGR2HSV);

				vector<Mat> hsv_db_planes;
				split(db_hsv_img, hsv_db_planes);

				// calculate the hue and saturation histograms for database image
				Mat hist_db_img_hue = makeHistogram(hsv_db_planes[0], histSize_hue, ranges_hue);
				Mat hist_db_img_sat = makeHistogram(hsv_db_planes[1], histSize_sat, ranges_sat);
				// calculate the accumulated hue and saturation histograms for database image
				Mat accumulatedHist_db_img_hue = accumulatedHist(hist_db_img_hue, h_bins);
				Mat accumulatedHist_db_img_sat = accumulatedHist(hist_db_img_sat, s_bins);
				// calculate the bins similarity in both source and database image's hue and saturation histogram
				int diff_in_hue_hist_bins = abs(accumulatedHist_src_hue.at<float>(h_bins - 1) - accumulatedHist_db_img_hue.at<float>(h_bins - 1));
				int diff_in_sat_hist_bins = abs(accumulatedHist_src_sat.at<float>(s_bins - 1) - accumulatedHist_db_img_sat.at<float>(s_bins - 1));
				// add the score 
				score[db_id] = compareHist(hist_src_hue, hist_db_img_hue, 0);
				score[db_id] += compareHist(hist_src_sat, hist_db_img_sat, 0);
				score[db_id] -= 0.1*diff_in_hue_hist_bins;
				score[db_id] -= 0.1*diff_in_sat_hist_bins;
				// Compute max score
				if (score[db_id] > 0 & labelCheck[db_id]) {
					if (score[db_id] > maxscore)
					{
						maxscore = score[db_id];
						maxscore_num = db_id;
						memcpy(maximg_name, tempname, 200 * sizeof(char));
					}
				}
				db_id++;
			}
		}
		fclose(fp);
	}
	else {
		printf("Loading label data for the database image ......\n");
		////* reading data Code copied from this c++ website , found at:
		////* http://www.cplusplus.com/doc/tutorial/files/
		string line;
		while (getline(labelFile, line))
		{
			stringstream labelData(line);
			labelData >> label[db_id];
				if (src_label == label[db_id]) {
					labelCheck[db_id] = true;
				}
				else {
					labelCheck[db_id] = false;
				}
				db_id++;
		}

		db_id = 0;
		char imagepath[200];
		fopen_s(&fp, IMAGE_LIST_FILE, "r");
		printf("Loading database images ......\n");
		while (!feof(fp))
		{
			while (fscanf_s(fp, "%s ", imagepath, sizeof(imagepath)) > 0)
			{
				printf("%s\n", imagepath);
				char tempname[200];
				sprintf_s(tempname, 200, "../%s", imagepath);

				db_img = imread(tempname); // read database image
				if (!db_img.data)
				{
					printf("Cannot find the database image number %d!\n", db_id);
					system("pause");
					return -1;
				}
				// save the database images that have the same labels with the specific source input image to a specific file
				if (labelCheck[db_id]) {
					string img_name = to_string(db_id);
					imwrite(path + img_name + ".jpg", db_img);
				}

				// convert the database image to hsv color and split its color channels
				Mat db_hsv_img;
				cvtColor(db_img, db_hsv_img, COLOR_BGR2HSV);

				vector<Mat> hsv_db_planes;
				split(db_hsv_img, hsv_db_planes);

				// calculate the hue and saturation histograms for database image
				Mat hist_db_img_hue = makeHistogram(hsv_db_planes[0], histSize_hue, ranges_hue);
				Mat hist_db_img_sat = makeHistogram(hsv_db_planes[1], histSize_sat, ranges_sat);
				// calculate the accumulated hue and saturation histograms for database image
				Mat accumulatedHist_db_img_hue = accumulatedHist(hist_db_img_hue, h_bins);
				Mat accumulatedHist_db_img_sat = accumulatedHist(hist_db_img_sat, s_bins);
				// calculate the bins similarity in both source and database image's hue and saturation histogram
				int diff_in_hue_hist_bins = abs(accumulatedHist_src_hue.at<float>(h_bins - 1) - accumulatedHist_db_img_hue.at<float>(h_bins - 1));
				int diff_in_sat_hist_bins = abs(accumulatedHist_src_sat.at<float>(s_bins - 1) - accumulatedHist_db_img_sat.at<float>(s_bins - 1));
				// add the score for correlation and decrease the score from difference in bins similarity
				score[db_id] = compareHist(hist_src_hue, hist_db_img_hue, 0);
				score[db_id] += compareHist(hist_src_sat, hist_db_img_sat, 0);
				score[db_id] -= 0.1*diff_in_hue_hist_bins;
				score[db_id] -= 0.1*diff_in_sat_hist_bins;
				// Compute max score
				if (score[db_id] > 0) {
					if (score[db_id] > maxscore & labelCheck[db_id])
					{
						maxscore = score[db_id];
						maxscore_num = db_id;
						memcpy(maximg_name, tempname, 200 * sizeof(char));
					}
				}
				db_id++;
			}
		}
		fclose(fp);
	}

	// show the best matching image 
	Mat maximg = imread(maximg_name);
	imshow("Best Match Image", maximg);
	printf("the most similar image is %d, the max score is %f\n", maxscore_num, maxscore);
	printf("Done \n");

	// Wait for the user to press a key in the GUI window.
	//Press ESC to quit
	int keyValue = 0;
	while (keyValue >= 0)
	{
		keyValue = cvWaitKey(0);

		switch (keyValue)
		{
		case 27:keyValue = -1;
			break;
		}
	}

    return 0;
}
