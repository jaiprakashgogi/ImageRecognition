#include "everything.h"

using namespace std;

mlp_t* mlp_create(int input, int hidden, int classes) {
    mlp_t* ret = new mlp_t();

    ret->num_input = input;
    ret->num_hidden = hidden;
    ret->num_output = classes;
    ret->dropout = 0.0f;
    ret->noise = 0.0f;
    ret->w_input = (float*)malloc(sizeof(float)*input*hidden);
    ret->w_hidden = (float*)malloc(sizeof(float)*hidden*classes);
    ret->bias_input = (float*)malloc(sizeof(float)*hidden);
    ret->bias_hidden = (float*)malloc(sizeof(float)*classes);

    return ret;
}

void mlp_initialize(mlp_t* mlp, cv::Mat data) {
    // Setup the weights from input -> hidden
    const float input_norm_mean = sqrtf(0.8f * (mlp->num_hidden + 1));

    float data_norm_mean = 0.0f;
    for (int i = 0;i<data.rows;i++) {
        cv::Scalar t = cv::sum(data.row(i));
        data_norm_mean += t[0] / data.cols;
    }
                        
    
    const float input_scale = 1.0f/(input_norm_mean*data_norm_mean);
    float w_input_lower = -0.5f*input_scale, w_input_upper=0.5f*input_scale;
    for(int i=0;i<mlp->num_hidden;i++) {
        mlp->w_input[i] = ((float)rand() / RAND_MAX)*(w_input_upper-w_input_lower) + w_input_lower;
    }

    cv::Mat mean = feature_mean(data);

    // Setup the weights from hidden -> output
    const float hidden_scale = 1/sqrtf(0.8f * (mlp->num_output + 1));
    float w_hidden_lower = -0.5f*hidden_scale, w_hidden_upper=0.5f*hidden_scale;
    for(int i=0;i<mlp->num_output;i++) {
        mlp->w_hidden[i] = ((float)rand() / RAND_MAX)*(w_hidden_upper-w_hidden_lower) + w_hidden_lower;
    }


    // Setup the biases from input -> hidden
    memset(mlp->bias_input, 0, sizeof(float)*mlp->num_hidden);

    // Setup biases from hidden -> output
    memset(mlp->bias_hidden, 0, sizeof(float)*mlp->num_output);
}

void mlp_dropout(mlp_t* mlp, float dropout) {
    mlp->dropout = dropout;
}


void mlp_noise(mlp_t* mlp, float noise) {
    mlp->noise = noise;
}

void mlp_corrupt_data(mlp_t* mlp, cv::Mat data, cv::Mat corrupted) {
    const int feature_size = data.cols;
    if(mlp->noise > 0) {
        if(((double)rand()/RAND_MAX) >= mlp->noise) {
            for(int i=0;i<feature_size;i++)
                corrupted.at<float>(i) = data.at<float>(i);
        } else {
            //printf("Corrupting data!\n");
        }
    } else {
        for(int i=0;i<feature_size;i++)
            corrupted.at<float>(i) = data.at<float>(i);
    }
}

inline float mlp_sigmoid(float y) {
    return 1.0f/(1.0f + expf(-y));
}

void mlp_forward_pass(mlp_t* mlp, cv::Mat input_y, cv::Mat hidden_y, cv::Mat data_row) {
    for(int i=0;i<mlp->num_hidden;i++) {
        if(((double)rand() / RAND_MAX) > mlp->dropout) {
            float y = data_row.dot(input_y) + mlp->input_bias.at<float>(0, i) / 32.0f;
            input_y.at<float>(0, i) = mlp_sigmoid(y);
        } else {
            // Drop this
            input_y.at<float>(0, i) = 0.0f;
        }
    }
}

void mlp_train(mlp_t* mlp, cv::Mat data, vector<uchar> labels, float ir, float hr, int start_epoch, int end_epoch, int max_epoch) {
    
    const int num_samples = data.rows;
    const int feature_size = data.cols;
    assert(num_samples == labels.size());
    int epoch = start_epoch+1;

    cv::Mat indexes = cv::Mat(num_samples, 1, CV_32FC1);
    for(int i=0;i<num_samples;i++) {
        indexes.at<float>(i, 0) = i;
    }

    // Randomize the order in which we read the data
    cv::Mat rand_indexes = shuffle_rows(indexes);

    const int batch_size = 32;
    cv::Mat input_y = cv::Mat::zeros(batch_size, mlp->num_hidden, CV_32FC1);
    cv::Mat hidden_y = cv::Mat::zeros(batch_size, mlp->num_output, CV_32FC1);

    do {
        // Do forward and backprop
        for(int i=0;i<num_samples/32;i++) {
            for(int j=0;j<batch_size;j++) {
                int idx = (int)indexes.at<float>(i*32+j);
                cv::Mat data_row = data.row(idx).clone();
                cv::Mat post_corruption = cv::Mat::zeros(1, feature_size, CV_32FC1);
                mlp_corrupt_data(mlp, data_row, post_corruption);

                mlp_forward_pass(mlp, input_y.row(j), hidden_y.row(j), post_corruption);
            }
        }
    } while(epoch++ < end_epoch);
}
