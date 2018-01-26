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
    erode(gray, gray, Mat(),Point(-1,-1), 1);
    //过滤离群点
    Mat kernel = getStructuringElement(MORPH_RECT, Size(10,10));
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
//    cout << " ratioBox = "<< ratioBox<<endl;
    if(ratioBox<scaleThresh){
        return false;
    }
//    cout<< " ratioBox = "<< ratioBox<<"======================"<<endl;
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

vector<int> imgprocess::detectionInBkg(vector<Detection> &detections, Mat mask) {
    cout<<"detectionInBkg  start"<<endl;
    vector<int> BkgDetectionIndex;
    for (int i = 0; i < detections.size(); ++i) {
        if (!isDynamicDetection(detections[i].getRect(), mask)){
            BkgDetectionIndex.push_back(i);
        } else{
            continue;
        }
    }
    cout << "detectionInBkg end"<<endl;
    return BkgDetectionIndex;
}

vector<int> imgprocess::detectionInFor(vector<Detection> &detections, Mat mask) {
    cout<<"detectionInBkg  start"<<endl;
    vector<int> BkgDetectionIndex;
    for (int i = 0; i < detections.size(); ++i) {
        if (isDynamicDetection(detections[i].getRect(), mask)){
            BkgDetectionIndex.push_back(i);
        } else{
            continue;
        }
    }
    cout << "detectionInBkg end"<<endl;
    return BkgDetectionIndex;
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

float imgprocess::isMatch(Detection detection1, Detection detection2, float iouThresh) {
//    cout<< " class name: " << detection1.getClass() << detection2.getClass()<<endl;
    if (detection1.getClass()!=detection2.getClass()) {
//        cout << "class name 不等" << endl;
        return 0;
    }
    float iou = getIOU(detection1.getRect(), detection2.getRect());
    if (iou<iouThresh)
        return 0;
//    cout<< " iou == " << iou << endl;
    return iou;
}

void imgprocess::addDetectionsWithoutWrongBoxInBkg(Mat imgBefore, Mat imgAfter, vector<Detection> &detectionBefor,
                                                   vector<Detection> &detectionAfter) {
    // detect images
    vector<Detection> resultBefore = tmp_detector.detect(imgBefore,0.7,0.4);
    vector<Detection> resultAfter = tmp_detector.detect(imgAfter,0.7,0.4);

    Mat img10 = imgBefore.clone();
    Mat img20 = imgAfter.clone();
    tmp_detector.drawBox(img10,resultBefore);
    tmp_detector.drawBox(img20,resultAfter);
    Size dsize = Size(720, 480);
    resize(img10, img10, dsize);
    resize(img20, img20, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/resultBefore"+std::to_string(frameNum)+".jpg", img10);
    imwrite("/home/wurui/Desktop/fugui/test/resultAfter"+std::to_string(frameNum)+".jpg", img20);

    // 寻找在bkg中的box: 在result中的序号
//    cout<< "寻找在bkg中的box: 在result中的序号" << endl;
    vector<int> BkgIndexBefore = detectionInBkg(resultBefore, maskTmp);
    vector<int> BkgIndexAfter = detectionInBkg(resultAfter, maskTmp);

    // debug 画出在bkg中的box
    vector<Detection> boxBkgB;
    vector<Detection> boxBkgA;
    for (int i = 0; i < BkgIndexBefore.size(); ++i) {
        boxBkgB.push_back(resultBefore[BkgIndexBefore[i]]);
    }
    for (int i = 0; i < BkgIndexAfter.size(); ++i) {
        boxBkgA.push_back(resultAfter[BkgIndexAfter[i]]);
    }
    Mat img1 = imgBefore.clone();
    Mat img2 = imgAfter.clone();
    Mat mask = maskTmp.clone();
    tmp_detector.drawBox(img1,boxBkgB);
    tmp_detector.drawBox(img2,boxBkgA);
    resize(img1, img1, dsize);
    resize(img2, img2, dsize);
    resize(mask, mask, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/boxBkgBefore"+std::to_string(frameNum)+".jpg", img1);
    imwrite("/home/wurui/Desktop/fugui/test/boxBkgAfter"+std::to_string(frameNum)+".jpg", img2);
    imwrite("/home/wurui/Desktop/fugui/test/maskFilter"+std::to_string(frameNum)+".jpg", mask);


    // iou 匹配, 去除配对失败且在bkg中的box
    float iouThresh = 0.4;
    // 配对成功后的detectionInBkg的标识为true
    map<int,bool> index_match_B;// <在result中的序号， 匹配标志>
    map<int,bool> index_match_A;
    for (int k = 0; k < BkgIndexBefore.size(); ++k) {
        index_match_B.insert(std::pair<int,bool>(BkgIndexBefore[k], false));
    }
    for (int k = 0; k < BkgIndexAfter.size(); ++k) {
        index_match_A.insert(std::pair<int,bool>(BkgIndexAfter[k], false));
    }
    int num = 0;
    for (int i = 0; i < BkgIndexBefore.size(); ++i) {
        if(index_match_B[BkgIndexBefore[i]]) continue; // 已配对过的detection跳过
        // 寻找iou最大的同名box
        int indexMatch = -1;
        float maxIOU = 0;

        for (int j = 0; j < BkgIndexAfter.size(); ++j) {
            if(index_match_A[BkgIndexAfter[j]]) continue; // 已配对过的detection跳过
            float iouTemp = isMatch(resultBefore[BkgIndexBefore[i]], resultAfter[BkgIndexAfter[j]],iouThresh);
//            cout<< "iouTmp = " << iouTemp<<"  maxIOU = " << maxIOU <<endl;
            if(iouTemp>=maxIOU){
                maxIOU = iouTemp;
                indexMatch = j;
//                cout<< "indexMatch j ="<<indexMatch<<endl;
            }
        }

        if (indexMatch>-1 && maxIOU>0){
            // 配对成功
            num++;
//            cout << " 配对成功: iou = " << maxIOU<<endl;
            index_match_B[BkgIndexBefore[i]]= true;
            index_match_A[BkgIndexAfter[indexMatch]] = true;
        }
    }
    cout << " 配对成功: 数量 = " << num<<endl;
    vector<Detection> boxNotMatchB;
    vector<Detection> boxNotMatchA;
    for (int i = 0; i < resultBefore.size(); ++i) {
        // 先判断 i 是否在 index_match_B 中
        bool isContain=false;
        for (int j = 0; j < BkgIndexBefore.size(); ++j) {
            if (i==BkgIndexBefore[j]) isContain = true;
        }
        if (isContain){
            if (!index_match_B[i]) {
                boxNotMatchB.push_back(resultBefore[i]);
                continue;
            }
        }
        detectionBefor.push_back(resultBefore[i]);
    }
    for (int i = 0; i < resultAfter.size(); ++i) {
        bool isContain=false;
        for (int j = 0; j < BkgIndexAfter.size(); ++j) {
            if (i==BkgIndexAfter[j]) isContain = true;
        }
        if(isContain){
            if (!index_match_A[i]) {
                boxNotMatchA.push_back(resultAfter[i]);
                continue;
            }
        }
        detectionAfter.push_back(resultAfter[i]);
    }
//    for (int i = 0; i < index_match_B.size(); ++i) {
//        boxNotMatchB.push_back(resultBefore[index_match_B[i]]);
//    }
//    for (int i = 0; i < index_match_A.size(); ++i) {
//        boxNotMatchA.push_back(resultAfter[index_match_A[i]]);
//    }

    // debug 画出匹配失败的box
    Mat img11 = imgBefore.clone();
    Mat img22 = imgAfter.clone();
    tmp_detector.drawBox(img11,boxNotMatchB);
    tmp_detector.drawBox(img22,boxNotMatchA);
    resize(img11, img11, dsize);
    resize(img22, img22, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/boxNotMatchBefore"+std::to_string(frameNum)+".jpg", img11);
    imwrite("/home/wurui/Desktop/fugui/test/boxNotMatchAfter"+std::to_string(frameNum)+".jpg", img22);
    frameNum++;
}

void imgprocess::addDetectionsWithoutWrongBoxInBkg2(Mat imgBefore, Mat imgAfter, vector<Detection> &detectionBefor,
                                                   vector<Detection> &detectionAfter) {
    // detect images
    vector<Detection> resultBefore0 = tmp_detector.detect(imgBefore,0.7,0.4);
    vector<Detection> resultAfter0 = tmp_detector.detect(imgAfter,0.7,0.4);
    cout<< " 检测到目标数量 before:"<< resultBefore0.size() << " after:"<<resultAfter0.size()<<endl;
    // debug 画所有检测box
    Mat img10 = imgBefore.clone();
    Mat img20 = imgAfter.clone();
    tmp_detector.drawBox(img10,resultBefore0);
    tmp_detector.drawBox(img20,resultAfter0);
    Size dsize = Size(720, 480);
    resize(img10, img10, dsize);
    resize(img20, img20, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/resultBefore"+std::to_string(frameNum)+".jpg", img10);
    imwrite("/home/wurui/Desktop/fugui/test/resultAfter"+std::to_string(frameNum)+".jpg", img20);
    Mat mask = maskTmp.clone();
    resize(mask, mask, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/maskFilter"+std::to_string(frameNum)+".jpg", mask);


    // iou 删除重叠过大的box
    vector<Detection> resultBefore = deleteBoxHighIOU(resultBefore0, 0.6);
    vector<Detection> resultAfter = deleteBoxHighIOU(resultAfter0, 0.6);
    // debug 画图
    Mat img1 = imgBefore.clone();
    Mat img2 = imgAfter.clone();
    tmp_detector.drawBox(img1,resultBefore);
    tmp_detector.drawBox(img2,resultAfter);
//    Size dsize = Size(720, 480);
    resize(img1, img1, dsize);
    resize(img2, img2, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/resultNewBefore"+std::to_string(frameNum)+".jpg", img1);
    imwrite("/home/wurui/Desktop/fugui/test/resultNewAfter"+std::to_string(frameNum)+".jpg", img2);


    // iou 匹配, 去除配对失败且在bkg中的box
    // 注意：iou匹配时不同名字之间的box不能匹配: 雪碧挡住营养快线，拿走雪碧， 会把雪碧和营养快线匹配，留下被遮挡的营养快线，而被遮挡的营养快线不在变化区，统计出错
    //      iou同类匹配剩下的box，完全在背景里的是误检测
    float iouThresh = 0.2;
    // 配对成功后的detectionInBkg的标识为true
    map<int,bool> index_match_B;// <在result中的序号， 匹配标志>
    map<int,bool> index_match_A;
    for (int k = 0; k < resultBefore.size(); ++k) {
        index_match_B.insert(std::pair<int,bool>(k, false));
    }
    for (int k = 0; k < resultAfter.size(); ++k) {
        index_match_A.insert(std::pair<int,bool>(k, false));
    }
    int num=0;
    for (int i = 0; i < resultBefore.size(); ++i) {
        if(index_match_B[i]) continue; // 已配对过的detection跳过
        // 寻找iou最大的同名box
        int indexMatch = -1;
        float maxIOU = 0;
        for (int j = 0; j < resultAfter.size(); ++j) {
            if(index_match_A[j]) continue; // 已配对过的detection跳过
            float iouTemp = isMatch(resultBefore[i], resultAfter[j],iouThresh);
//            cout<< "iouTmp = " << iouTemp<<"  maxIOU = " << maxIOU <<endl;
            if(iouTemp>=maxIOU){
                maxIOU = iouTemp;
                indexMatch = j;
//                cout<< "indexMatch j ="<<indexMatch<<endl;
            }
        }

//        cout<<"MaxIOU = "<<maxIOU<<endl;
        if (indexMatch>-1 && maxIOU>0){
            // 配对成功
            num++;
//            cout << " 配对成功: iou = " << maxIOU<<" before index: "<<i<<" after index: "<< indexMatch <<endl;
            index_match_B[i]= true;
            index_match_A[indexMatch] = true;
        }
    }
    cout<<"配对成功数量： "<<num<< endl;
    vector<Detection> boxNotMatchB;
    vector<Detection> boxNotMatchA;
    for (int i = 0; i < resultBefore.size(); ++i) {
        if(!index_match_B[i]){
            boxNotMatchB.push_back(resultBefore[i]);
            continue;
        }
//        detectionBefor.push_back(resultBefore[i]);
    }
    for (int i = 0; i < resultAfter.size(); ++i) {
        if(!index_match_A[i]) {
            boxNotMatchA.push_back(resultAfter[i]);
            continue;
        }
//        detectionAfter.push_back(resultAfter[i]);//匹配成功的都加入result
    }
    // debug 画出匹配失败的box
    Mat img11 = imgBefore.clone();
    Mat img22 = imgAfter.clone();
    tmp_detector.drawBox(img11,boxNotMatchB);
    tmp_detector.drawBox(img22,boxNotMatchA);
    resize(img11, img11, dsize);
    resize(img22, img22, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/boxNotMatchBefore"+std::to_string(frameNum)+".jpg", img11);
    imwrite("/home/wurui/Desktop/fugui/test/boxNotMatchAfter"+std::to_string(frameNum)+".jpg", img22);

    // 在匹配失败的box中： 属于变化区域的，加入result中 等待sku统计; 属于背景区域的 当作虚检忽略
    // 寻找在变化区中的box: 在result中的序号
//    cout<< "匹配失败的box中寻找在变化区中的box" << endl;
    vector<int> IndexBefore = detectionInFor(resultBefore, maskTmp);
    vector<int> IndexAfter = detectionInFor(resultAfter, maskTmp);
    for (int i = 0; i < IndexBefore.size(); ++i) {
        index_match_B[IndexBefore[i]]= true;
    }
    for (int i = 0; i < IndexAfter.size(); ++i) {
        index_match_A[IndexAfter[i]]=true;
    }
    vector<Detection> boxLeftBefore;
    vector<Detection> boxLeftAfter;
    for (int i = 0; i < resultBefore.size(); ++i) {
        if(!index_match_B[i]){
            boxLeftBefore.push_back(resultBefore[i]);
            continue;
        }
        detectionBefor.push_back(resultBefore[i]);
    }
    for (int i = 0; i < resultAfter.size(); ++i) {
        if(!index_match_A[i]) {
            boxLeftAfter.push_back(resultAfter[i]);
            continue;
        }
        detectionAfter.push_back(resultAfter[i]);//匹配成功的都加入result
    }
    // debug 画出最后剩下的box
    Mat img111 = imgBefore.clone();
    Mat img222 = imgAfter.clone();
    tmp_detector.drawBox(img111,boxLeftBefore);
    tmp_detector.drawBox(img222,boxLeftAfter);
    resize(img111, img111, dsize);
    resize(img222, img222, dsize);
    imwrite("/home/wurui/Desktop/fugui/test/boxLeftBefore"+std::to_string(frameNum)+".jpg", img111);
    imwrite("/home/wurui/Desktop/fugui/test/boxLeftAfter"+std::to_string(frameNum)+".jpg", img222);

    frameNum++;
}

vector<Detection> imgprocess::deleteBoxHighIOU(vector<Detection> detections, float iouThresh) {
    map<int, bool> indexes;
    vector<Detection> newDetection;
//    cout << "detection num =========== "<<detections.size()<<endl;
    for (int i = 0; i < detections.size(); ++i) {
        indexes.insert(std::pair<int, bool>(i,true));
    }
    for (int i = 0; i < detections.size(); ++i) {
        if(!indexes[i]) continue; // 这个box已经被删除
        for (int j = 0; j < detections.size(); ++j) {
            if(!indexes[j]) continue;
            if(i==j) continue;
            float iou = getIOU(detections[i].getRect(), detections[j].getRect());
//            if(iou>0) cout << " iou : " << iou<<endl;
            if (iou<iouThresh) continue;
            int toDelete = detections[i].getScore()<detections[j].getScore()? i:j;
            cout << " 被删除的box："<< toDelete<< detections[toDelete].getClass() << "  "<<detections[toDelete].getRect()<<" i="<<i<<" j="<<j<<endl;
            indexes[toDelete]=false;
        }
    }
    for (int k = 0; k < detections.size(); ++k) {
        if(!indexes[k]) continue;
        newDetection.push_back(detections[k]);
    }
//    cout<< " new detection num ===="<<newDetection.size()<<endl;
    return newDetection;
}