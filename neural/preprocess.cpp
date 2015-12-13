#include "everything.h"

cv::Mat normalize(cv::Mat patches, float epsilon) {
    cv::Mat ret;
    patches.convertTo(ret, CV_32FC1);

    const int num_rows = patches.rows;

    //printf("Standardizing all patches\n");
    // TODO does this need to be per-channel? I believe now.

#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0;i<num_rows;i++) {
        cv::Mat current = ret.row(i);
        cv::Mat temp2;

        cv::Scalar t = cv::mean(current);
        float mean = t[0];
        cv::pow(current-mean, 2, temp2);
        t = cv::sum(temp2)/(3*PATCH_SIZE*PATCH_SIZE);
        float var = t[0];
        float sd = sqrtf(var + epsilon);

        // Varies from about -1 to 1
        ret.row(i) = (current - mean) / sd;
    }

    return ret;
}

cv::Mat extract_patches(cv::Mat img, int patch_size) {
    const int num_rows = img.rows;
    const int num_cols = img.cols;

    const int num_patch_rows = num_rows - patch_size;
    const int num_patch_cols = num_cols - patch_size;

    cv::Mat patches = cv::Mat::zeros(num_patch_rows * num_patch_cols, patch_size*patch_size*3, CV_8UC1);
    int idx = 0;
    for(int y=0;y<num_patch_rows;y++) {
        for(int x=0;x<num_patch_cols;x++) {
            cv::Rect r(x, y, patch_size, patch_size);
            cv::Mat temp = img(r).clone();

            cv::Mat aux = patches.row(idx);
            temp.reshape(1, 1).copyTo(aux);
            idx++;
        }
    }

    return patches;
}

cv::Mat zca_white(cv::Mat data, cv::Mat mean, cv::Mat whitener) {
    cv::Mat avg = feature_mean(data);

    avg.copyTo(mean);

    const int num_vectors = data.rows;
    const int feature_size = data.cols;
    cv::Mat ret = data.clone();

    for(int i=0;i<num_vectors;i++) {
        ret.row(i) -= avg;
    }

    cv::Mat tr_data;
    cv::transpose(data, tr_data);
    cv::Mat sigma = (tr_data * data) / feature_size;
    cv::SVD svd(sigma, cv::SVD::FULL_UV);
    
    cv::Mat U, Vt, S;
    svd.compute(sigma, S, U, Vt);
    S = cv::Mat::diag(S);
    
    const float epsilon = 0.1f;
    cv::Mat t3_temp = S + cv::Mat::eye(S.rows, S.cols, CV_32FC1) * epsilon;
    cv::Mat t3_temp2;
    cv::invert(t3_temp, t3_temp2);
    cv::Mat t3;
    cv::sqrt(t3_temp2, t3);

    cv::Mat t4;
    cv::transpose(U, t4);

    const float t1 = sqrtf(num_vectors)-1;
    cv::Mat whitening_transform;
    whitening_transform = t1 * U * t3 * t4;

    ret = ret * whitening_transform;
    whitening_transform.copyTo(whitener);


    return ret;
}


