//
// Created by wurui on 18-1-23.
//

#include "imgprocess.h"

using namespace std;
using namespace cv;

imgprocess::imgprocess() {}
imgprocess* imgprocess::myinstance = NULL;
imgprocess* imgprocess::getInstance() {
    if(myinstance==NULL){
        myinstance = new imgprocess();
    }
    return myinstance;
}

void imgprocess::init(string labelfile, string protofile, string modelfile) {
    labelFile = labelfile;
    protoFile = protofile;
    modelFile = modelfile;
    std::vector<std::string> voc_classes;
    ifstream label_file(labelFile);
    while (!label_file.eof()) {
        string label_name;
        label_file >> label_name;
        if(label_name.empty())
            continue;
        voc_classes.push_back(label_name);
        imgprocess::myinstance->skuName.push_back(label_name);
    }
    tmp_detector.init(protoFile, modelFile, voc_classes);
    tmp_detector.setComputeMode("gpu", 0);
}

Mat imgprocess::pointFilter(Mat gray) {
    //过滤离群点，找凸包
    medianBlur(gray, gray, 5);
    vector<Point> points;
    for (int r = 0; r < gray.rows; r++) {
        for (int c = 0; c < gray.cols; c++) {
            if (gray.at<uchar>(r, c) != 0) {
                points.push_back(Point(c, r));
            }
        }
    }
    vector<int> hull;
    convexHull(Mat(points), hull, true);

    if(hull.empty())
    {
        return gray;
    }
    int hullcount = (int)hull.size();
    Point pt0 = points[hull[hullcount - 1]];

    for(int i = 0; i < hullcount; i++ )
    {
        Point pt = points[hull[i]];
        line(gray, pt0, pt, Scalar(0, 255, 255), 1, CV_AA);
        pt0 = pt;
    }

    return gray;
}

Mat imgprocess::imgDiff2(Mat framepro, Mat frame) {
    Mat gray,dframe;
    if(framepro.size != frame.size) cerr << "image size not match: two imgs come from different cam?" <<endl;
    cvtColor(framepro, framepro, CV_BGR2GRAY);
    cvtColor(frame, frame, CV_BGR2GRAY);
    absdiff(frame, framepro, dframe);//帧间差分计算两幅图像各个通道的相对应元素的差的绝对值。
    threshold(dframe, dframe, 80, 255, CV_THRESH_BINARY);//阈值分割
    // 膨胀腐蚀
    gray = dframe.clone();
    dilate(gray, gray, Mat(),Point(-1,-1), 3);
    erode(gray, gray, Mat(),Point(-1,-1), 3);
    //过滤离群点
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    morphologyEx(gray,gray, MORPH_OPEN, kernel, Point(-1,-1));
    return gray;
}

Mat imgprocess::imgMog2(Mat img1, Mat img2) {
    Mat mask;
    //MOG2
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(100,50,false);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    // MOG2
    for (int i = 0; i < 100; ++i) {
        pMOG2->apply(img1,mask);
    }
    pMOG2->apply(img2,mask);
    //对处理后的帧进行开操作，减少视频中较小的波动造成的影响
    morphologyEx(mask,mask, MORPH_OPEN, kernel, Point(-1,-1));
    return mask;
}

bool imgprocess::isDynamicDetection(cv::Rect box, Mat mask, float scaleThresh) {
    int area = box.height*box.height;
    int count = 0;//前景像素数量
//    scaleThresh=0.2;
    for (int i = box.x; i < box.width+box.x; ++i) {
        for (int j = box.y; j < box.height+box.y; ++j) {
            int value = mask.at<uchar>(j,i);//(row,col)
            if(value>125){
                count++;
            }
        }
    }
    float ratioBox = (float)count / area;
    cout << " ratioBox = "<< ratioBox<<endl;
    if(ratioBox<scaleThresh){
        return false;
    }
    cout<< " ratioBox = "<< ratioBox<<"======================"<<endl;
    return true;
}

bool imgprocess::isDynamicCamera(Mat imgBefore, Mat imgAfter,float scaleThresh) {
    cout << " is dynamic : ==========="<< imgAfter.size << " " << imgBefore.size  << endl;
    Mat mask = this->imgDiff2(imgBefore, imgAfter);
    cout << "mask == "<< mask.size << endl;
    int area = mask.cols*mask.rows;
    int count = 0;
    for (int i = 0; i < mask.cols; ++i) {
        for (int j = 0; j < mask.rows; ++j) {
            int value = mask.at<uchar>(j,i);
            if (value>125) count++;// 变化像素
        }
    }
    float ratio = (float)count / area;
    cout << " ratio = "<< ratio<<endl;
    if(ratio<scaleThresh){
        cout << " scale area < thresh ================"<< endl;
        return false;
    }
    cout << mask.size << " ===============mask size"<<endl;
    maskTmp = NULL;
    maskTmp = mask.clone();
    cout << maskTmp.size << "================mask tmp"<<endl;
//    cout << "mask:  " << maskTmp.size<<endl;
    return true;
}

vector<Detection> imgprocess::detectionBkgFilt(vector<Detection> &detections, vector<Detection> &detectionsNew, Mat mask) {
    vector<Detection> DynamicDetection;
    for (int i = 0; i < detections.size(); ++i) {
        if(isDynamicDetection(detections[i].getRect(), mask)){
            cout << " dynamic box :"<< detections[i].getClass() << endl;
            detectionsNew.push_back(detections[i]);
            DynamicDetection.push_back(detections[i]);
        } else{
            continue;
        }
    }
    return DynamicDetection;
}

void imgprocess::addDynamicDetections(Mat imgBefore, Mat imgAfter, vector<Detection> &detectionBefore,
                                       vector<Detection> &detectionAfter) {
    // detect images
    vector<Detection> resultBefore = tmp_detector.detect(imgBefore,0.7,0.4);
    vector<Detection> resultAfter = tmp_detector.detect(imgAfter,0.7,0.4);

    // detections filter
    vector<Detection> DynamicDetectionB = detectionBkgFilt(resultBefore, detectionBefore, maskTmp);
    vector<Detection> DynamicDetectionA = detectionBkgFilt(resultAfter, detectionAfter, maskTmp);

    // show debug
    Mat img1 = imgBefore.clone();
    Mat img2 = imgAfter.clone();
    Mat mask = maskTmp.clone();
    tmp_detector.drawBox(img1,DynamicDetectionB);
    tmp_detector.drawBox(img2,DynamicDetectionA);
    tmp_detector.drawBox(imgBefore, resultBefore);
    tmp_detector.drawBox(imgAfter, resultAfter);
    Size dsize = Size(720, 480);
    resize(imgBefore, imgBefore, dsize);
    resize(imgAfter, imgAfter, dsize);
    resize(img1, img1, dsize);
    resize(img2, img2, dsize);
    resize(mask, mask, dsize);
    imshow("debug before", imgBefore);
    imshow("debug after", imgAfter);
    imwrite("/home/wurui/Desktop/fugui/debugB"+std::to_string(frameNum)+".jpg", imgBefore);
    imwrite("/home/wurui/Desktop/fugui/debugA"+std::to_string(frameNum)+".jpg", imgAfter);
    imwrite("/home/wurui/Desktop/fugui/debugdynameicB"+std::to_string(frameNum)+".jpg", img1);
    imwrite("/home/wurui/Desktop/fugui/debugdynamicA"+std::to_string(frameNum)+".jpg", img2);
    imwrite("/home/wurui/Desktop/fugui/maskFilter"+std::to_string(frameNum)+".jpg", mask);
    frameNum++;
    waitKey(50);
}

float imgprocess::getIOU(cv::Rect box1, cv::Rect box2) {
    int w = max(box1.x+box1.width, box2.x+box2.width) - min(box1.x, box2.x);
    int h = max(box1.y+box1.height, box2.y+box2.height) - min(box1.y, box2.y);
    if((box1.width+box2.width-w)<=0 || (box1.height+box2.height-h)<=0)
        return 0;
    float area = (float)((box1.width+box2.width-w)*(box1.height+box2.height-h));
    float iou = area/(float)(box1.width*box1.height + box2.height*box2.width - area);
    return iou;
}

void imgprocess::deleteBoxLowIouInBkg(vector<Detector> &detectionsB, vector<Detection> &detectionsA) {
    vector<int> indexB;// 待删除的编号
    vector<int> indexA;
    vector<Detection> boxB; boxB.assign(detectionsB.begin(), detectionsB.end());
    vector<Detection> boxA; boxA.assign(detectionsA.begin(), detectionsA.end());
    for (int i = 0; i < ; ++i) {

    }
}