#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videostab.hpp>
#include <vector>
#include <sys/stat.h>
#include <omp.h>

#include <stdlib.h>
#include <string.h>

#include "knobs.h"

/////////////////////////////////////////////////////////
// Timing functions
// Record the execution time of some code, in milliseconds.
#define DECLARE_TIMING(s)  int64 timeStart_##s; double timeDiff_##s; double timeTally_##s = 0; int countTally_##s = 0
#define START_TIMING(s)    timeStart_##s = cvGetTickCount()
#define STOP_TIMING(s)     timeDiff_##s = (double)(cvGetTickCount() - timeStart_##s); timeTally_##s += timeDiff_##s; countTally_##s++
#define GET_TIMING(s)      (double)(timeDiff_##s / (cvGetTickFrequency()*1000.0))
#define GET_AVERAGE_TIMING(s)   (double)(countTally_##s ? timeTally_##s/ ((double)countTally_##s * cvGetTickFrequency()*1000.0) : 0)
#define CLEAR_AVERAGE_TIMING(s) timeTally_##s = 0; countTally_##s = 0

////////////////////////////////////////////////////////
// Structs


////////////////////////////////////////////////////////
// Function prototypes

// learn.cpp
cv::Mat load_data(char* filename);
void zca_learn(cv::Mat *mean, int mean_j, cv::Mat *u, const cv::Mat data, float epsilon);
int main(int argc, char* argv[]);

// utils.cpp
int getMemValue();
cv::Mat shuffle_rows(const cv::Mat &matrix);
bool file_exists (const std::string& name);
cv::Mat feature_mean(cv::Mat);

// visualize.cpp
cv::Mat visualize_patches(cv::Mat, cv::Mat);
cv::Mat visualize_patches_std(cv::Mat, cv::Mat);
cv::Mat visualize_patches_zca(cv::Mat, cv::Mat);
cv::Mat visualize_kmeans_centroids(cv::Mat);
