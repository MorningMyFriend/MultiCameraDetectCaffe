//
// Created by wurui on 18-1-11.
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

void detect(){
    //    // sku class
    vector<string> skuName; //= {"cola","xuebi","weitaNingmeng","zhenguoli_blue","zhenguoli_red","redbull","chapi","kangshifu","laotangsuancai","tangdaren","heweidao" };
    map<string,int> skuBefore;
    map<string,int> skuAfter;
    map<string,int> skuNameTaken;

    // detector init
    Detector tmp_detector;
    std::vector<std::string> voc_classes;
    ifstream label_file("/home/wurui/Desktop/fugui/FUGUI/predefined_classes.txt");
    while (!label_file.eof()) {
        string label_name;
        label_file >> label_name;
        if(label_name.empty())
            continue;
        voc_classes.push_back(label_name);
        skuName.push_back(label_name);
        skuNameTaken[label_name]=0;
        skuBefore[label_name] = 0;
        skuAfter[label_name] = 0;
        cout << label_name << " : " <<skuBefore[label_name] << " " << skuAfter[label_name] <<endl;
    }
    tmp_detector.init("/home/wurui/Desktop/fugui/FUGUI/test.prototxt",
                      "/home/wurui/Desktop/fugui/FUGUI/FuGui20180122topview_iter_100000.caffemodel", voc_classes);
    tmp_detector.setComputeMode("gpu", 0);


    for (int i = 0; i < 8; i++) {
        Mat img=imread("/home/wurui/Desktop/fugui/shot/test/shot"+std::to_string(i)+".png");
        vector<Detection> resultBefore = tmp_detector.detect(img,0.7,0.4);
        tmp_detector.drawBox(img, resultBefore);
        Size dsize = Size(720,480);
        resize(img, img, dsize);
        imwrite("/home/wurui/Desktop/fugui/shot/test/result"+std::to_string(i)+".png",img);
    }
}

void videoShot(){
    // detect video shot
    string videoPath = "/home/wurui/Desktop/fugui/data/test4.avi";
    cv::VideoCapture mycap(videoPath);
    if (mycap.isOpened()){
        cout << " video ok "<< endl;
    }

    int shotCount = 0;
    while (1) {
        cv::Mat img;
        mycap >> img;
        cv::imshow("video", img);
        cv::waitKey(50);
        int key = cv::waitKey(50);
        if (key > 0) {
            cv::imwrite("/home/wurui/Desktop/fugui/shot/test4/shot" + std::to_string(shotCount) + ".png", img);
            shotCount++;
            cout << shotCount << " shot count " << endl;
        }
    }
}
int main(){
    videoShot();
    cout<< "no thing to do"<<endl;
}
//
//struct cameraImg {
//    Mat imgBefore;
//    Mat imgAfter;
//};
//
//bool isGoodDetection(cv::Rect box, Mat mask){
//    int area = box.height*box.height;
//    int count = 0;//前景像素数量
//    int scaleThresh=0.2;
//    for (int i = box.x; i < box.width+box.x; ++i) {
//        for (int j = box.y; j < box.height+box.y; ++j) {
//            int value = mask.at<uchar>(j,i);//(row,col)
//            if(value>125){
//                count++;
//            }
//        }
//    }
//    if(count/area<scaleThresh){
//        return false;
//    }
//    return true;
//}
//
//vector<Detection> detectionsBkgFiltered(vector<Detection> detections,Mat mask,map<string,int> skuNameTaken,vector<string> skuname){
//    vector<Detection> detectionsNew;
//    for (int i = 0; i < skuname.size() ; ++i) {
//        if (skuNameTaken[skuname[i]]<1){
//            continue;
//        }
//        string sku = skuname[i];
//        for (int j = 0; j < detections.size(); ++j) {
//            if(detections[j].getClass()!=sku) continue;
//            if(isGoodDetection(detections[j].getRect(), mask)){
//                detectionsNew.push_back(detections[j]);
//            } else{
//                continue;
//            }
//        }
//    }
//    return detectionsNew;
//}
//
//int main()
//{
//    // sku class
//    vector<string> skuName; //= {"cola","xuebi","weitaNingmeng","zhenguoli_blue","zhenguoli_red","redbull","chapi","kangshifu","laotangsuancai","tangdaren","heweidao" };
//    map<string,int> skuBefore;
//    map<string,int> skuAfter;
//    map<string,int> skuNameTaken;
//
//    // detector init
//    Detector tmp_detector;
//    std::vector<std::string> voc_classes;
//    ifstream label_file("/home/wurui/project/fugui/models/1-9/predefined_classes.txt");
//    while (!label_file.eof()) {
//        string label_name;
//        label_file >> label_name;
//        if(label_name.empty())
//            continue;
//        voc_classes.push_back(label_name);
//        skuName.push_back(label_name);
//        skuNameTaken[label_name]=0;
//        skuBefore[label_name] = 0;
//        skuAfter[label_name] = 0;
//        cout << label_name << " : " <<skuBefore[label_name] << " " << skuAfter[label_name] <<endl;
//    }
//    tmp_detector.init("/home/wurui/project/fugui/models/1-9/deploy.prototxt",
//                      "/home/wurui/project/fugui/models/1-9/fugui_20180109_iter_90000.caffemodel", voc_classes);
//    tmp_detector.setComputeMode("gpu", 0);
//
//
//    // detect video shot
//    string videoPath = "/home/wurui/Desktop/fugui/test3.avi";
//    cv::VideoCapture mycap(videoPath);
//    if (mycap.isOpened()){
//        cout << " video ok "<< endl;
//    }
//
//    int shotCount = 0;
//    while (1) {
//        cv::Mat img;
//        mycap >> img;
//        cv::imshow("video", img);
//        cv::waitKey(10);
//        int key = cv::waitKey(10);
//        if(key >0 ){
//            cv::imwrite("/home/wurui/Desktop/fugui/shot"+std::to_string(shotCount)+".png",img);
//            shotCount++;
//            cout << shotCount << " shot count "<< endl;
//        }
//        if(shotCount>1){
//            shotCount = 0;
//            // img differe process
//            cv::Mat imgBefore = cv::imread("/home/wurui/Desktop/fugui/shot0.png");
//            cv::Mat imgAfter = cv::imread("/home/wurui/Desktop/fugui/shot1.png");
//            imgprocess imgprocess1 = imgprocess();
//            Mat mask = imgprocess1.imgDiff2(imgBefore, imgAfter);
//            imshow("maskDiff",mask);
//            waitKey(50);
//
//            // detect images
//            vector<Detection> resultBefore = tmp_detector.detect(imgBefore,0.5);
//            vector<Detection> resultAfter = tmp_detector.detect(imgAfter,0.5);
//
//            for(int i=0;i<resultBefore.size();i++){
//                string className = resultBefore[i].getClass();
//                skuBefore[className] += 1;
//            }
//            for(int i=0;i<resultAfter.size();i++){
//                string className = resultAfter[i].getClass();
//                skuAfter[className] += 1;
//            }
//
//            for(int i=0;i<skuName.size();i++) {
//                int countBefore = skuBefore[skuName[i]];
//                cout << "skuBefore ---------------------------" << endl;
//                cout << skuName[i] <<" : " << countBefore << endl;
//            }
//
//            for(int i=0;i<skuName.size();i++) {
//                int countAfter = skuAfter[skuName[i]];
//                cout << "skuAfter ---------------------------" << endl;
//                cout << skuName[i] << " : " << countAfter << endl;
//            }
//
//            cout << "=========================== result =================================" << endl;
//            for(int i=0;i<skuName.size();i++){
//                int countBefore = skuBefore[skuName[i]];
//                int countAfter = skuAfter[skuName[i]];
//                if (countBefore != countAfter){
//                    skuNameTaken[skuName[i]]=1;
//                    cout << skuName[i] << " take off: " << countBefore-countAfter << endl;
//                } else{
//                    cout << skuName[i] << " take off: 0 " << endl;
//                }
//            }
//
//            // bkg filter wrong detection
//            vector<Detection> resultDetection = detectionsBkgFiltered(resultAfter,mask,skuNameTaken,skuName);
//            Mat imgFilt = imgAfter.clone();
//
//            tmp_detector.drawBox(imgBefore,resultBefore);
//            tmp_detector.drawBox(imgAfter,resultAfter);
//            tmp_detector.drawBox(imgFilt,resultDetection);
//            Size dsize = Size(720,480);
//            resize(imgBefore,imgBefore,dsize);
//            cv::imshow("before",imgBefore);
//            cv::imwrite("imgBefore.jpg", imgBefore);
//            resize(imgAfter,imgAfter,dsize);
//            cv::imshow("after",imgAfter);
//            cv::imwrite("imgAfter.jpg", imgAfter);
//            resize(imgFilt,imgFilt,dsize);
//            cv::imshow("bkgFilt",imgFilt);
//            cv::imwrite("imgFilt.jpg", imgFilt);
//            cv::waitKey(1000);
//        }
//    }
//
////    // img differe process
////    cv::Mat imgBefore = cv::imread("/home/wurui/Desktop/fugui/shot0.png");
////    cv::Mat imgAfter = cv::imread("/home/wurui/Desktop/fugui/shot1.png");
////    imgprocess imgprocess1 = imgprocess();
////    Mat mask = imgprocess1.imgDiff2(imgBefore, imgAfter);
////    imshow("maskDiff",mask);
////    waitKey(50);
////
////    // detect images
////    vector<Detection> resultBefore = tmp_detector.detect(imgBefore,0.5);
////    vector<Detection> resultAfter = tmp_detector.detect(imgAfter,0.5);
////
////    for(int i=0;i<resultBefore.size();i++){
////        string className = resultBefore[i].getClass();
////        skuBefore[className] += 1;
////    }
////    for(int i=0;i<resultAfter.size();i++){
////        string className = resultAfter[i].getClass();
////        skuAfter[className] += 1;
////    }
////
////    for(int i=0;i<skuName.size();i++) {
////        int countBefore = skuBefore[skuName[i]];
////        cout << "skuBefore ---------------------------" << endl;
////        cout << skuName[i] <<" : " << countBefore << endl;
////    }
////
////    for(int i=0;i<skuName.size();i++) {
////        int countAfter = skuAfter[skuName[i]];
////        cout << "skuAfter ---------------------------" << endl;
////        cout << skuName[i] << " : " << countAfter << endl;
////    }
////
////    cout << "=========================== result =================================" << endl;
////    for(int i=0;i<skuName.size();i++){
////        int countBefore = skuBefore[skuName[i]];
////        int countAfter = skuAfter[skuName[i]];
////        if (countBefore != countAfter){
////            skuNameTaken[skuName[i]]=1;
////            cout << skuName[i] << " take off: " << countBefore-countAfter << endl;
////        } else{
////            cout << skuName[i] << " take off: 0 " << endl;
////        }
////    }
////
////    // bkg filter wrong detection
////    vector<Detection> resultDetection = detectionsBkgFiltered(resultAfter,mask,skuNameTaken,skuName);
////    Mat imgFilt = imgAfter.clone();
////
////    tmp_detector.drawBox(imgBefore,resultBefore);
////    tmp_detector.drawBox(imgAfter,resultAfter);
////    tmp_detector.drawBox(imgFilt,resultDetection);
////    Size dsize = Size(720,540);
////    resize(imgBefore,imgBefore,dsize);
////    cv::imshow("before",imgBefore);
////    cv::imwrite("imgBefore.jpg", imgBefore);
////    resize(imgAfter,imgAfter,dsize);
////    cv::imshow("after",imgAfter);
////    cv::imwrite("imgAfter.jpg", imgAfter);
////    resize(imgFilt,imgFilt,dsize);
////    cv::imshow("bkgFilt",imgFilt);
////    cv::imwrite("imgFilt.jpg", imgFilt);
////    cv::waitKey(0);
//
//}
