#include "everything.h"

using namespace std;

cv::Mat zca_whitening(cv::Mat data, cv::Mat mean, int mean_j, cv::Mat u) {
    int j=0;
    const int num_samples = data.rows;
    const int sample_size = data.cols;

    cv::Mat ret = cv::Mat(data.rows, data.cols, CV_32FC1);

#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for(j=0;j<num_samples;j++) {
        float mval = mean.at<float>(0, j);
        cv::Mat sub = data.row(j) - mval;

        //printf("Size of data row = %d*%d\n", data.row(j).rows, data.row(j).cols);
        //printf("Size of sub row = %d*%d\n", sub.rows, sub.cols);
        //printf("Size of u row = %d*%d\n", u.rows, u.cols);

        for(int i=0;i<sample_size;i++) {
            cv::Scalar total = cv::sum(sub.at<float>(1, i) * u.row(i));
            ret.at<float>(j, i) = total[0];
        }

        if(j%10000 == 0) 
            printf("Done\n");
        //data.row(j) = sub.mul(u.reshape(1, 1));
        //nv_matrix_t *sub = nv_matrix_alloc(x->n, 1);
    
        // ZCAWhitend_x = (x - mean) * U
        // nv_vector_sub(sub, 0, x, x_j, mean, mean_j);
        //nv_vector_mulmtr(x, x_j, sub, 0, u);
        //nv_matrix_free(&sub);
    }

    return ret;
}

void zca_learn(cv::Mat *mean, int mean_j, cv::Mat *u, const cv::Mat data, float epsilon) {
    int num_samples = data.rows;
    int sample_size = data.cols;

    cv::Mat cov, cov_mean;

DECLARE_TIMING(cov_timer);
START_TIMING(cov_timer);
    cv::calcCovarMatrix(data, cov, cov_mean, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_32F);
STOP_TIMING(cov_timer);

    printf("Cov matrix is %d*%d\n", cov.rows, cov.cols);
    printf("cov_mean is %d*%d\n", cov_mean.rows, cov_mean.cols);
    printf("data is %d*%d\n", data.rows, data.cols);
    printf("Calculating the cov took %d ms\n", (int)GET_TIMING(cov_timer));

    cv::Mat eigenvectors, eigenvalues;

DECLARE_TIMING(eig_timer);
START_TIMING(eig_timer);
    cv::eigen(cov, eigenvalues, eigenvectors);
STOP_TIMING(eig_timer);

    printf("Done calculating eigenvectors");
    printf("Calculation too %d ms\n", (int)GET_TIMING(eig_timer));

    int sz[2];
    sz[0] = sample_size; sz[1] = sample_size;
    cv::Mat d = cv::Mat(2, sz, CV_32FC1);
    cv::Mat v = cv::Mat(2, sz, CV_32FC1);

    for(int i=0;i<sample_size;i++) {
        d.at<float>(i, i) = sqrtf(1.0f / (eigenvalues.at<float>(i) + epsilon));
    }

    v = eigenvectors * d;
    cout << "Size of v = " << v.size() << endl;
    cout << "Size of eigenvectors = " << eigenvectors.size() << endl;
    cout << "Size of d = " << d.size() << endl;

    cv::Mat tr_eigenvectors;
    cv::transpose(eigenvectors, tr_eigenvectors);
    *u = v * tr_eigenvectors;

    cv::Mat tr;
    cv::transpose(*u, tr);
    tr.copyTo(*u);

    cov_mean.copyTo(*mean);
}

cv::Mat standardize(cv::Mat patches, float epsilon) {
    cv::Mat ret;
    patches.convertTo(ret, CV_32FC1);

    const int num_rows = patches.rows;

    printf("Standardizing all patches\n");
    // TODO does this need to be per-channel? I believe now.

#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0;i<num_rows;i++) {
        cv::Mat current = ret.row(i);
        cv::Mat temp, temp2;

        current.convertTo(temp, CV_32FC1);
    
        cv::Scalar t = cv::mean(current);
        float mean = t[0];
        cv::pow(temp-mean, 2, temp2);
        t = cv::sum(temp2)/(3*PATCH_SIZE*PATCH_SIZE);
        float var = t[0];
        float sd = sqrtf(var + epsilon);

        temp = (temp - mean); // sd;
        temp.convertTo(current, CV_8UC1);
        //current = current / sd;
    }

    return ret;
}

cv::Mat load_data(char* filename, vector<cv::Mat> &images) {
    FILE *fp;
    int size;

    printf("Opening dataset: %s\n", filename);
    // Ensure the file can be loaded
    fp = fopen(filename, "rb");
    if(!fp) {
        printf("The file does not exist: %s\n", filename);
        return cv::Mat();
    }

    // Find the number of elements in this
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);

    // Ensure we have an exact division
    if(size % (NUM_BYTES_PER_IMG+1) != 0) {
        printf("Non-integer number of elements in the given file. Exitting.\n");
        return cv::Mat();
    }

    //const int num_samples = size / (NUM_BYTES_PER_IMG + 1);
    const int num_samples = 100;
    const int size_minus_patch = (IMG_SIZE-PATCH_SIZE) * (IMG_SIZE-PATCH_SIZE);
    const int rcount = PATCH_SIZE * PATCH_SIZE * 3;
    cv::Mat data = cv::Mat::zeros(size_minus_patch*num_samples, rcount, CV_8UC1);
    printf("Number of images in dataset = %d\n", num_samples);


    // Rewind to the beginning to start reading actual data
    fseek(fp, 0, SEEK_SET);

    // Loop through the entire dataset
    int sample_count = 0;
    int sz[3] = {IMG_SIZE*PATCH_SIZE*PATCH_SIZE, IMG_SIZE-PATCH_SIZE, IMG_SIZE-PATCH_SIZE};
    cv::Mat patches = cv::Mat(3, sz, CV_32FC1, cv::Scalar(0));
    cv::Mat *scratch_channel = new cv::Mat[3];
    cv::Mat scratch;

    DECLARE_TIMING(file_reader);
    START_TIMING(file_reader);
    DECLARE_TIMING(individual_reader);

    int data_idx = 0;
    while(sample_count < num_samples) {
        START_TIMING(individual_reader);

        // Read in the data
        uchar label;
        uchar *img_data = new uchar[NUM_BYTES_PER_IMG];

        // Read label for this sample
        if(fread(&label, sizeof(uchar), 1, fp) != 1) {
            printf("Unable to read label for image %d\n", sample_count);
            return cv::Mat();
        }

        // Read pixel data
        if(fread(img_data, sizeof(uchar), NUM_BYTES_PER_IMG, fp) != NUM_BYTES_PER_IMG) {
            printf("Unable to read image data for image %d\n", sample_count);
            return cv::Mat();
        }

        // Generate a proper OpenCV matrix
        scratch_channel[0] = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC1, img_data+0);
        scratch_channel[1] = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC1, img_data+1024);
        scratch_channel[2] = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC1, img_data+2048);
        cv::merge(scratch_channel, 3, scratch);
        images.push_back(scratch);

        // Calculate the individual patches
        int start_idx = data_idx;
        for(int y=0;y<IMG_SIZE-PATCH_SIZE;y++) {
            for(int x=0;x<IMG_SIZE-PATCH_SIZE;x++) {
                cv::Rect r(x, y, PATCH_SIZE, PATCH_SIZE);
                cv::Mat temp = scratch(r).clone();

                cv::Mat aux = data.row(data_idx);
                scratch(r).clone().reshape(1, 1).copyTo(aux);
                data_idx++;
            }
        }

        // Visualize the patches generated by the above code
        //cv::Mat visual = visualize_patches(scratch, data.rowRange(start_idx, data_idx));
        //cv::imshow("Visualizing patches", visual);
        //cv::waitKey(0);


        // Print progress
        if(sample_count%1000 == 0){
            printf("Sample count = %d\n", sample_count);
            printf("Reading %d took %d ms\n", sample_count, (int)GET_AVERAGE_TIMING(file_reader));
        }
        sample_count += 1;
        STOP_TIMING(individual_reader);
    }
    STOP_TIMING(file_reader);
    printf("It took %d ms to read the entire data file\n", (int)GET_TIMING(file_reader));

    fclose(fp);
    //return shuffle_rows(data);
    return data;
}

int main(int argc, char* argv[]) {
    const int rcount = PATCH_SIZE * PATCH_SIZE * 3;
    cv::Mat data = cv::Mat(rcount, SAMPLES, CV_32FC1);
    cv::Mat centroids = cv::Mat(rcount, CENTROIDS, CV_32FC1);
    cv::Mat zca_u = cv::Mat(rcount, rcount, CV_32FC1);
    cv::Mat zca_m = cv::Mat(rcount, 1, CV_32FC1);
    vector<cv::Mat> images;

    // Reading this from a file takes longer than executing this.
    printf("Memory usage initially: %d KB\n", getMemValue());
    cv::Mat patches = load_data("../data/data_batch_1.bin", images);
    printf("Number of images loaded = %ld\n", (long)images.size());
    printf("Memory usage after loading images: %d KB\n", getMemValue());

    cv::Mat patches_std = standardize(patches, STANDARDIZE_EPSILON);
    printf("Memory usage: %d KB\n", getMemValue());

    cv::Mat visual = visualize_patches(images[0], patches.rowRange(0, 676));
    cv::Mat visual_std = visualize_patches(images[0], 255+patches_std.rowRange(0, 676));
    cv::imshow("Visualizing patches", visual);
    cv::imshow("Visualizing patches - standardized", visual_std);
    cv::waitKey(0);

    printf("Starting calculating the covariance matrix\n");
    zca_learn(&zca_m, 0, &zca_u, patches, 0.1f);
    printf("Memory usage after zca learning: %d KB\n", getMemValue());

    printf("Size of zca_u = %d*%d\n", zca_u.rows, zca_u.cols);
    cv::Mat patches_whitened = zca_whitening(patches_std, zca_m, 0, zca_u);

    printf("Number of rows = %d\n", patches.rows);
    printf("Memory usage after whitening: %d KB\n", getMemValue());
    return 0;
}
