aux_source_directory(./ source_files)
add_library(intelligentCabinetTools SHARED
        ${source_files})
target_link_libraries(intelligentCabinetTools
        caffe
        ${OpenCV_LIBS}
        glog
        boost_system)