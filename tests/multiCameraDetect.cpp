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

using namespace std;

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

            // 储存可视化结果
//            Size dsize = Size(720, 480);
//            imgprocess::getInstance()->tmp_detector.drawBox(imgs[2*i], DetectionsBefore);
//            imgprocess::getInstance()->tmp_detector.drawBox(imgs[2*i+1], DetectionsAfter);
//            resize(imgs[2*i],imgs[2*i],dsize);
//            resize(imgs[2*i+1],imgs[2*i+1],dsize);
//            cout << " mask =="<<endl;
//            resize(imgprocess::getInstance()->maskTmp,imgprocess::getInstance()->maskTmp,dsize);
//            imwrite("/home/wurui/Desktop/fugui/beforeDynamic"+std::to_string(i)+".jpg", imgs[2*i]);
//            imwrite("/home/wurui/Desktop/fugui/afterDynamic"+std::to_string(i)+".jpg", imgs[2*i+1]);
//            imwrite("/home/wurui/Desktop/fugui/maskDynamic"+std::to_string(i)+".jpg", imgprocess::getInstance()->maskTmp);
//            imshow("before", imgs[2*i]);
//            imshow("after", imgs[2*i]+1);
//            imshow("mask", imgprocess::getInstance()->maskTmp);
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
    // 找图像变化的相机, 将所有变化区域的Detections
    vector<Detection> DetectionsBefore;
    vector<Detection> DetectionsAfter;
    int cameraNum = imgs.size()/2;
//    imgprocess
    for (int i = 0; i < cameraNum; ++i) {
        cout << " i== " << i << " img size: " << imgs[2 * i + 1].size << " imgs nums = " << imgs.size() << endl;
        if (imgprocess::getInstance()->isDynamicCamera(imgs[2 * i], imgs[2 * i + 1])) {
            imgprocess::getInstance()->addDetectionsWithoutWrongBoxInBkg(imgs[2 * i], imgs[2 * i + 1], DetectionsBefore,
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
};


int main(){
    // 一次传输 2n 张图像；n 个相机
    vector<Mat> imgs;
    for (int i = 0; i < 8; i++) {
        Mat img=imread("/home/wurui/Desktop/fugui/shot/test/shot"+std::to_string(i)+".png");
        imgs.push_back(img);
    }

    // 入口
    string label_file = "/home/wurui/Desktop/fugui/FUGUI/predefined_classes.txt";
    string net_prototxt = "/home/wurui/Desktop/fugui/FUGUI/test.prototxt";
    string model_file = "/home/wurui/Desktop/fugui/FUGUI/FuGui20180122topview_iter_100000.caffemodel";

    imgprocess::getInstance()->init(label_file, net_prototxt, model_file);
    cout << "init ok ==========  "<<endl;
    map<string,int> skuTaken = skuDetect2(imgs);

    return 0;
}



