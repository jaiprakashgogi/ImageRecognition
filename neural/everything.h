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
struct training_t {
    cv::Mat data;
    std::vector<uchar> labels;
    unsigned int num_samples;
};

struct mlp_t {
    int num_input;
    int num_hidden;
    int num_output;
    float dropout;
    float noise;
    float *w_input;
    float *w_hidden;
    float *bias_input;
    float *bias_hidden;
};

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
void writeMatFile(const char *fileName, cv::Mat mat, const char* name);

// Preprocessing.cpp
cv::Mat normalize(cv::Mat, float);
cv::Mat extract_patches(cv::Mat img, int patch_size);
cv::Mat zca_white(cv::Mat data, cv::Mat mean, cv::Mat whitener);

// visualize.cpp
cv::Mat visualize_patches(cv::Mat, cv::Mat);
cv::Mat visualize_patches_std(cv::Mat, cv::Mat);
cv::Mat visualize_patches_zca(cv::Mat, cv::Mat);
cv::Mat visualize_kmeans_centroids(cv::Mat);

// mlp.cpp
mlp_t* mlp_create(int, int, int);
void mlp_initialize(mlp_t*, cv::Mat);
void mlp_dropout(mlp_t* mlp, float dropout);
void mlp_noise(mlp_t* mlp, float noise);
void mlp_train(mlp_t*, cv::Mat, std::vector<uchar>, float, float, int, int, int);
