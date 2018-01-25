//
// Created by wurui on 18-1-23.
//

#ifndef INTELLIGENTCABINET_IMGPROCESS_H
#define INTELLIGENTCABINET_IMGPROCESS_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "detector.h"
#include "detection.h"

using namespace std;
using namespace cv;

class imgprocess {
private:
    string labelFile;
    string protoFile;
    string modelFile;
    int frameNum = 0;

public:
    static imgprocess *myinstance;
    static imgprocess *getInstance();

    imgprocess();

    Detector tmp_detector;
    vector<string> skuName;
    Mat maskTmp;

    void init(string labelfile, string protofile, string modelfile);

    Mat imgDiff2(Mat framepro, Mat frame);
    Mat pointFilter(Mat gray);
    Mat imgMog2(Mat img1, Mat img2);

    bool isDynamicDetection(cv::Rect box, Mat mask, float scaleThresh=0.01);
    bool isDynamicCamera(Mat imgBefore, Mat imgAfter, float scaleThresh=0.01);

    vector<Detection> detectionBkgFilt(vector<Detection> &detections, vector<Detection> &detectionsNew, Mat mask);
    void addDynamicDetections(Mat imgBefore, Mat imgAfter,
                               vector<Detection> &detectionBefore, vector<Detection> &detectionAfter);

};

#endif //INTELLIGENTCABINET_IMGPROCESS_H
