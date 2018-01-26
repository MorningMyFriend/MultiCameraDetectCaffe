add_executable(test_detect pva_detect_test.cpp)
target_link_libraries(test_detect
        intelligentCabinetTools)

add_executable(skuDetect skuDetect.cpp imgprocess.cpp imgprocess.h)
target_link_libraries(skuDetect
        intelligentCabinetTools)

add_executable(cameraDetect  imgprocess.cpp imgprocess.h multiCameraDetect.cpp)
target_link_libraries(cameraDetect
        intelligentCabinetTools -lpthread)

