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

using namespace std;

std::vector<std::string> readFileList(char *basePath)
{
    std::vector<std::string> result;
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
            continue;
        else if(ptr->d_type == 8)    ///file
        {printf("d_name:%s/%s\n",basePath,ptr->d_name);
            result.push_back(std::string(ptr->d_name));}
        else if(ptr->d_type == 10)    ///link file
        {printf("d_name:%s/%s\n",basePath,ptr->d_name);
            result.push_back(std::string(ptr->d_name));}
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            result.push_back(std::string(ptr->d_name));
            readFileList(base);
        }
    }
    closedir(dir);
    return result;
}


int main()
{
//    std::vector<std::string> fileNames = readFileList("/home/zt/Desktop/fuguiPic/");
    Detector tmp_detector;
    std::vector<std::string> voc_classes;
    ifstream label_file("/home/wurui/project/fugui/models/1-9/predefined_classes.txt");
    while (!label_file.eof()) {
        string label_name;
        label_file >> label_name;
        if(label_name.empty())
            continue;
        voc_classes.push_back(label_name);
    }

    tmp_detector.init("/home/wurui/project/fugui/models/1-9/deploy.prototxt",
                      "/home/wurui/project/fugui/models/1-10/fugui_simple_iter_90000.caffemodel", voc_classes);
    tmp_detector.setComputeMode("gpu", 0);

    string videoPath = "/home/wurui/Desktop/fugui/test3.avi";
    cv::VideoCapture mycap(videoPath);
    timeval tpstart, tpend;
//    for(int i = 0; i < fileNames.size(); ++i) {
    while(1) {
        cv::Mat img;// = cv::imread("/home/zt/Desktop/fuguiPic/" + fileNames[i]);
        mycap >> img;
        gettimeofday(&tpstart, NULL);
        std::vector<Detection> result = tmp_detector.detect(img, 0.5);
        gettimeofday(&tpend, NULL);
        LOG(ERROR) << "use time: " << (1000000 * (tpend.tv_sec - tpstart.tv_sec)
                                       + tpend.tv_usec - tpstart.tv_usec) / 1000000.0;

        tmp_detector.drawBox(img, result);

        cv::imshow("result", img);
        cv::waitKey(10);
    }
}
