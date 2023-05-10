#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat img, Gray, imgCanny, Thresh, Dil, Erode, Blur, imgWarp, imgCrop;
vector<Point> initialPoints, docPoints;
bool is_paused = false;


float w = 420, h = 596;

Mat preProcessing(Mat image)
{
	cvtColor(image, Gray, COLOR_BGR2GRAY);
	GaussianBlur(Gray, Blur, Size(3, 3), 3, 0);
	Canny(Blur, imgCanny, 25, 75);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, Dil, kernel);
	
	return Dil;

}

vector<Point> getContours(Mat image) {


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	int maxArea = 0;


		for (int i = 0; i < contours.size(); i++)
		{
			int area = contourArea(contours[i]);

			string objectType;

			if (area > 1000)
			{
				float peri = arcLength(contours[i], true);
				approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

				if (area > maxArea && conPoly[i].size() == 4)
				{
					biggest = { conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3] };
					maxArea = area;
				}
			}
		}
	return biggest;
}

void drawPoints(vector<Point> points, Scalar color)
{
	for (int i = 0; i < points.size(); i++)
	{
		circle(img, points[i], 10, color, FILLED);
		putText(img, to_string(i), points[i], FONT_HERSHEY_PLAIN, 4, color, 4);
	}
}

vector<Point> reorder(vector<Point> points)
{
	vector<Point> newPoints;
	vector<int>  sumPoints, subPoints;

	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //3

	return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
	Point2f src[4] = { points[0],points[1],points[2],points[3] };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	return imgWarp;
}

void main() {

	//string path = "Resources/paper.jpg";
	//img = imread(path);

	VideoCapture cap(0);


	Rect roi(5, 5, w - (2 * 5), h - (2 * 5));

	while (true)
	{
		cap.read(img);
		
		if (!is_paused)
		{
			imshow("image", img);
		}

		char c = (char)waitKey(25); // press SPACE to pause
		if (c == 32) 
		{
			is_paused = true;
		}

		if (is_paused)
		{
			cout << "Paused. Press SPACE to resume" << endl;

			Thresh = preProcessing(img);

			initialPoints = getContours(Thresh);
			docPoints = reorder(initialPoints);
			imgWarp = getWarp(img, docPoints, w, h);
			imgCrop = imgWarp(roi);
			
			imshow("document", imgCrop);
			
			c = (char)waitKey(0);
			if (c == 32) { is_paused = false; }

		}

		waitKey(1);
	}
}