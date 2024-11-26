cmake_minimum_required(VERSION 3.10)

project(rknn_yolo11_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_yolo11_file rknpu2/yolo11.cc)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    ${rknpu_yolo11_file}
)

add_library(yolo11 STATIC
    postprocess.cc
    ${rknpu_yolo11_file}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

target_link_libraries(yolo11
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
    target_link_libraries(yolo11 Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

target_include_directories(yolo11 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

# Currently zero copy only supports rknpu2, v1103/rv1103b/rv1106 supports zero copy by default
if (NOT (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103" OR TARGET_SOC STREQUAL "rk1808" 
    OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126" OR TARGET_SOC STREQUAL "rv1103b"))
    add_executable(${PROJECT_NAME}_zero_copy
        main.cc
        postprocess.cc
        rknpu2/yolo11_zero_copy.cc
    )

    add_library(yolo11_zero_copy STATIC
        postprocess.cc
        rknpu2/yolo11_zero_copy.cc
    )

    target_compile_definitions(${PROJECT_NAME}_zero_copy PRIVATE ZERO_COPY)
    target_compile_definitions(yolo11_zero_copy PRIVATE ZERO_COPY)

    target_link_libraries(${PROJECT_NAME}_zero_copy
        imageutils
        fileutils
        imagedrawing    
        ${LIBRKNNRT}
        dl
    )

    target_link_libraries(yolo11_zero_copy
        imageutils
        fileutils
        imagedrawing    
        ${LIBRKNNRT}
        dl
    )


    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${PROJECT_NAME}_zero_copy Threads::Threads)
    endif()

    target_include_directories(${PROJECT_NAME}_zero_copy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBRKNNRT_INCLUDES}
        ${LIBTIMER_INCLUDES}
    )

    target_include_directories(yolo11_zero_copy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBRKNNRT_INCLUDES}
        ${LIBTIMER_INCLUDES}
    )
    install(TARGETS ${PROJECT_NAME}_zero_copy DESTINATION .)
endif()

# install(TARGETS ${PROJECT_NAME} DESTINATION .)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
# file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
# install(FILES ${RKNN_FILES} DESTINATION model)