//
// Created by wurui on 18-1-24.
//
#include "detector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <iostream>
#include "glog/logging.h"
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <map>
#include "imgprocess.h"
#include <map>
#include "pthread.h"
#include <chrono>

using namespace chrono;
using namespace std;


bool isShot = false;
pthread_mutex_t lock;


struct threadParam {
    int threadId;
    vector<Mat> frames;
};

void *threadFrameShot(void *param) {
    threadParam *frames = (threadParam *) param;
    int id = frames->threadId;
    cout << "thread id : " << frames->threadId << endl;
    cout<<"frames num:"<<frames->frames.size()<<endl;

    // detect video shot
    string videoPath = "/home/wurui/Desktop/fugui/data/test"+std::to_string(id)+".avi";
    cv::VideoCapture mycap(videoPath);
    if (mycap.isOpened()) {
        cout << " video ok " << endl;
    }
    int shotCount = 0;
    while (1) {
        cv::Mat img;
        mycap >> img;
        cout<<"thread "<<id<<" img size"<< img.size<<endl;
        sleep(0.001);
        cv::imshow("video"+std::to_string(id), img);
        imwrite("/home/wurui/Desktop/"+std::to_string(id)+".png",img);
        cv::waitKey(0);
        if(isShot) {
            frames->frames.push_back(img);
            shotCount++;
        }
        if (shotCount==1) break;
        sleep(0.01);
    }
}

vector<Mat> multiCameraDetect(){
    pthread_t ps[2];
    pthread_mutex_init(&lock,NULL);
    int camNum =2;
    vector<Mat> frame1;
    vector<Mat> frame2;

    for (int i = 0; i < 2; ++i) {
        pthread_t p;
        ps[i] = p;
        threadParam param;
        switch (i){
            case 0:
                param.frames = frame1;
                break;
            case 1:
                param.frames = frame2;
                break;
        }
        param.threadId = i;
        pthread_create(&ps[i],NULL,threadFrameShot,&param);
    }
    int num=0;
    while(1){
        cout<<"main thread"<<endl;
        waitKey(10);
        int key = cv::waitKey(10);
        if (key > 0) {
            isShot = true;
            cout<<"main: isShot = true"<<endl;
            num++;
        }
        if(num==1) break;
        sleep(0.1);
        isShot=false;
        cout<<"main: isShot = false"<<endl;
    }
    cout<<"frame num:"<<frame1.size()<<" "<<frame2.size()<<endl;
    vector<Mat> frames;
    frames.push_back(frame1[0]);
    frames.push_back(frame1[1]);
    frames.push_back(frame2[0]);
    frames.push_back(frame2[1]);
}

map<string,int> skuDetect1(vector<Mat> imgs){
    // vector<Mat> imgs = { cam_1_before cam_1_after cam_2_before cam_2 after ...}
    // 检测图片数量是否正确
    if(imgs.size()%2!=0){
        cerr<< " images num invalid "<<endl;
        exit(0);
    }
    // 找图像变化的相机, 将所有变化区域的Detections
    vector<Detection> DetectionsBefore;
    vector<Detection> DetectionsAfter;
    int cameraNum = imgs.size()/2;
//    imgprocess
    for (int i = 0; i < cameraNum; ++i) {
        cout << " i== "<< i << " img size: " << imgs[2*i+1].size<< " imgs nums = " << imgs.size() << endl;
        if(imgprocess::getInstance()->isDynamicCamera(imgs[2*i],imgs[2*i+1])){
            // 这个相机的图像变化了
            imshow("input", imgs[2*i+1]);
            waitKey(30);
            imgprocess::getInstance()->addDynamicDetections(imgs[2*i],imgs[2*i+1],DetectionsBefore,DetectionsAfter);
        }

    }
    // 统计变化区域 sku 的拿取情况
    map<string,int> skuBefore;
    map<string,int> skuAfter;
    map<string,int> skuNameTaken;
    for (int j = 0; j < imgprocess::getInstance()->skuName.size(); ++j) {
        skuBefore.insert(std::pair<string,int>(imgprocess::getInstance()->skuName[j],0));
        skuAfter.insert(std::pair<string,int>(imgprocess::getInstance()->skuName[j],0));
    }
    for (int k = 0; k < DetectionsBefore.size(); ++k) {
        skuBefore[DetectionsBefore[k].getClass()]+=1;
    }
    for (int k = 0; k < DetectionsAfter.size(); ++k) {
        skuAfter[DetectionsAfter[k].getClass()]+=1;
    }
    cout << "============================= 结果统计： ==============================" << endl;
    for (int l = 0; l < imgprocess::getInstance()->skuName.size(); ++l) {
        if (skuAfter[imgprocess::getInstance()->skuName[l]] != skuBefore[imgprocess::getInstance()->skuName[l]]){
            skuNameTaken.insert(std::pair<string,int>(imgprocess::getInstance()->skuName[l],
                                         skuBefore[imgprocess::getInstance()->skuName[l]] - skuAfter[imgprocess::getInstance()->skuName[l]]));
            cout << imgprocess::getInstance()->skuName[l]<<" 被拿走: "<< skuNameTaken[imgprocess::getInstance()->skuName[l]]<<endl;
        }
    }
};

map<string,int> skuDetect2(vector<Mat> imgs){
    // vector<Mat> imgs = { cam_1_before cam_1_after cam_2_before cam_2 after ...}
    // 检测图片数量是否正确
    if(imgs.size()%2!=0){
        cerr<< " images num invalid "<<endl;
        exit(0);
    }
    cout<< " time start >>>>>>>>>>>"<< endl;
    auto time1 = system_clock::now();
    // 找图像变化的相机, 将所有变化区域的Detections
    vector<Detection> DetectionsBefore;
    vector<Detection> DetectionsAfter;
    int cameraNum = imgs.size()/2;
//    imgprocess
    for (int i = 0; i < cameraNum; ++i) {
//        cout << " i== " << i << " img size: " << imgs[2 * i + 1].size << " imgs nums = " << imgs.size() << endl;
        if (imgprocess::getInstance()->isDynamicCamera(imgs[2 * i], imgs[2 * i + 1])) {
            imgprocess::getInstance()->addDetectionsWithoutWrongBoxInBkg2(imgs[2 * i], imgs[2 * i + 1], DetectionsBefore,
                                                                         DetectionsAfter);
        }
    }
    // 统计变化区域 sku 的拿取情况
    map<string,int> skuBefore;
    map<string,int> skuAfter;
    map<string,int> skuNameTaken;
    for (int j = 0; j < imgprocess::getInstance()->skuName.size(); ++j) {
        skuBefore.insert(std::pair<string,int>(imgprocess::getInstance()->skuName[j],0));
        skuAfter.insert(std::pair<string,int>(imgprocess::getInstance()->skuName[j],0));
    }
    for (int k = 0; k < DetectionsBefore.size(); ++k) {
        skuBefore[DetectionsBefore[k].getClass()]+=1;
    }
    for (int k = 0; k < DetectionsAfter.size(); ++k) {
        skuAfter[DetectionsAfter[k].getClass()]+=1;
    }
    cout << "============================= 结果统计： ==============================" << endl;
    for (int l = 0; l < imgprocess::getInstance()->skuName.size(); ++l) {
        if (skuAfter[imgprocess::getInstance()->skuName[l]] != skuBefore[imgprocess::getInstance()->skuName[l]]){
            skuNameTaken.insert(std::pair<string,int>(imgprocess::getInstance()->skuName[l],
                                                      skuBefore[imgprocess::getInstance()->skuName[l]] - skuAfter[imgprocess::getInstance()->skuName[l]]));
            cout << imgprocess::getInstance()->skuName[l]<<" 被拿走: "<< skuNameTaken[imgprocess::getInstance()->skuName[l]]<<endl;
        }
    }


    auto time2 = system_clock::now();
    auto duration=duration_cast<microseconds>(time2-time1);
    cout << "time end ~~~~~"<<endl;
    cout<<"time use===== total 8:"<<(double)(duration.count())*microseconds::period::num / microseconds::period::den<<"s"<<endl;
    cout<<"time use===== total 8:"<<(double)(duration.count())*microseconds::period::num<<"ms"<<endl;
    cout<<"time use===== per img:"<<(double)(duration.count())*microseconds::period::num / 8<<"ms"<<endl;

//
    for (int m = 0; m < cameraNum; ++m) {
        Mat imgbefore = imread("/home/wurui/Desktop/fugui/test/resultNewBefore"+std::to_string(m)+".jpg");
        Mat imgafter = imread("/home/wurui/Desktop/fugui/test/resultNewAfter"+std::to_string(m)+".jpg");
        imshow("bofore camera:"+std::to_string(m),imgbefore);
        imshow("after camera:"+std::to_string(m),imgafter);
//        waitKey(0);
    }
    cv::waitKey(0);
};

int main(){
    // 一次传输 2n 张图像；n 个相机
    vector<Mat> imgs;
    vector<Mat> frameBefore;
    vector<Mat> frameAfter;

//    Mat img1 = imread("/home/wurui/Desktop/fugui/shot/test/shot10.png");
//    Mat img2 = imread("/home/wurui/Desktop/fugui/shot/test/shot12.png");
//    imgs.push_back(img1);
//    imgs.push_back(img2);

    // detect video shot
    int cameraNum = 4;
    string videoPaths[cameraNum];
    vector<cv::VideoCapture> mycap;
    for (int i = 0; i < cameraNum; ++i) {
        videoPaths[i] = "/home/wurui/Desktop/fugui/data/test"+std::to_string(i)+".avi";
        cv::VideoCapture mycapi(videoPaths[i]);
        mycap.push_back(mycapi);
    }

    cout<<"read video done"<<endl;
    int shotCount = 0;
    while (1) {
        if (shotCount>1) break;
        Mat frame[cameraNum];
        for (int i = 0; i < cameraNum; ++i) {
            Mat img;
            mycap[i] >> img;
            frame[i] = img;
            imshow("video"+std::to_string(i),img);
        }
        cv::waitKey(50);
        int key = cv::waitKey(50);
        if (key > 0) {
            for (int i = 0; i < cameraNum; ++i) {
                if (shotCount==0){
                    frameBefore.push_back(frame[i]);
                } else{
                    frameAfter.push_back(frame[i]);
                }
            }
            shotCount++;
            cout << shotCount << " shot count " << endl;
        }
    }
    for (int j = 0; j < cameraNum; ++j) {
        imgs.push_back(frameBefore[j]);
        imgs.push_back(frameAfter[j]);
    }
    cout<< "===================detect start==================="<<endl;

    // 入口
    string label_file = "/home/wurui/Desktop/fugui/FUGUI/predefined_classes.txt";
    string net_prototxt = "/home/wurui/Desktop/fugui/FUGUI/test.prototxt";
    string model_file = "/home/wurui/Desktop/fugui/FUGUI/FuGui20180122topview_iter_100000.caffemodel";

    imgprocess::getInstance()->init(label_file, net_prototxt, model_file);
    cout << "init ok ==========  "<<endl;

//    cout<< " time start >>>>>>>>>>>"<< endl;
//    auto time1 = system_clock::now();
    map<string,int> skuTaken = skuDetect2(imgs);
//    auto time2 = system_clock::now();
//    auto duration=duration_cast<microseconds>(time2-time1);
//    cout << "time end ~~~~~"<<endl;
//    cout<<"time use===== total 8:"<<(double)(duration.count())*microseconds::period::num / microseconds::period::den<<"s"<<endl;
//    cout<<"time use===== total 8:"<<(double)(duration.count())*microseconds::period::num<<"ms"<<endl;
//    cout<<"time use===== per img:"<<(double)(duration.count())*microseconds::period::num / 8<<"ms"<<endl;

    return 0;
}



