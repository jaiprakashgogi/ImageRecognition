#include "everything.h"

using namespace std;

cv::Mat extract_features(cv::Mat img, cv::Mat mean, cv::Mat whitener, cv::Mat centroids) {
    // This function updates the idx row of data
    cv::Mat patches = extract_patches(img, PATCH_SIZE);
    cv::Mat patches_norm = normalize(patches, STANDARDIZE_EPSILON);
    cv::Mat patches_zca = zca_white(patches_norm, mean, whitener);

    patches.release();
    patches_norm.release();

    cv::Mat pool = cv::Mat::zeros(CENTROIDS, 4, CV_32FC1);

    // Pooling from gigantic feature space into a 4*CENTROIDS space
    const int num_patches = patches_zca.rows;
    const int feature_size = patches_zca.cols;

#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0;i<num_patches;i++) {
        const int y = i / (IMG_SIZE-PATCH_SIZE);
        const int x = i % (IMG_SIZE-PATCH_SIZE);

        int pool_index;
        const int r = (int)sqrtf(POOL_GRID);
        int y_idx = (y / ((IMG_SIZE-PATCH_SIZE) / r));
        int x_idx = (x / ((IMG_SIZE-PATCH_SIZE) / r));

        // Did we go beyong the image boundary?
        if (x_idx >= r) {
            x_idx = r-1;
        }
        if (y_idx >= r) {
            y_idx = r-1;
        }

        pool_index = y_idx * r + x_idx;
        if (pool_index >= POOL_GRID) {
            pool_index = POOL_GRID - 1;
        }
        
        // TODO - implement triangle distance measure??
        // Using nearest neighbour for now

        cv::Mat current = patches_zca.row(i);
        cv::Mat distances = cv::Mat::zeros(1, CENTROIDS, CV_32FC1);

        int l = 0;
        for(l=0;l<CENTROIDS;l++) {
            cv::Scalar d = cv::norm(centroids.row(i) - current);
            distances.col(l) = d[0];
        }

        double minval = 0;
        cv::Point minloc;
        cv::minMaxLoc(distances, &minval, NULL, &minloc);
        const unsigned int idx = (unsigned int)minloc.x;

        distances.release();
        current.release();

        pool.at<float>(idx, pool_index) = 1.0f + pool.at<float>(idx, pool_index);
    }
    
    return pool.reshape(1, POOLED_FEATURE_SIZE).clone();
}

training_t load_data_neural(char* filename, bool generate_flips, cv::Mat mean, cv::Mat whitener, cv::Mat centroids) {
    FILE *fp;
    int size;

    printf("Opening dataset: %s\n", filename);
    fp = fopen(filename, "rb");
    if(!fp) {
        printf("Could not open file.\n");
        return training_t();
    }

    // Go to the end
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);

    // Rewind to the beginning
    fseek(fp, 0, SEEK_SET);

    if(size % (NUM_BYTES_PER_IMG+1) != 0) {
        printf("Non-integer number of elements in this file.\n");
        return training_t();
    }

    int num_samples = size / (NUM_BYTES_PER_IMG + 1);
    //int num_samples = 100;
    int num_samples_original = num_samples;

    // We'll be generating twice the number of samples
    if(generate_flips) {
        num_samples *= 2;
    }

    const int size_minus_patch = (IMG_SIZE-PATCH_SIZE) * (IMG_SIZE-PATCH_SIZE);
    const int rcount = PATCH_SIZE * PATCH_SIZE * 3;

    int sample_count = 0;

    training_t ret;
    ret.num_samples = num_samples;

    // 4000 feature vector for each image
    cv::Mat data = cv::Mat::zeros(num_samples, POOLED_FEATURE_SIZE, CV_32FC1);
    cv::Mat *scratch_channel = new cv::Mat[3]();
    cv::Mat scratch, scratch2;

    vector<cv::Mat> images;
    for(int i=0;i<num_samples_original;i++) {
        printf("Loading image #%d\n", sample_count);
        uchar label;
        uchar *img_data = new uchar[NUM_BYTES_PER_IMG];

        if(fread(&label, sizeof(uchar), 1, fp) != 1) {
            printf("Unable to read label for image %d\n", sample_count);
            return training_t();
        }

        if(fread(img_data, sizeof(uchar), NUM_BYTES_PER_IMG, fp) != NUM_BYTES_PER_IMG) {
            printf("Unable to read image data for image %d\n", sample_count);
            return training_t();
        }

        // Construct an OpenCV image
        scratch_channel[0] = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC1, img_data+0);
        scratch_channel[1] = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC1, img_data+1024);
        scratch_channel[2] = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC1, img_data+2048);
        cv::merge(scratch_channel, 3, scratch);

        // Process the image and extract features
        ret.labels.push_back(label);
        images.push_back(scratch.clone());

        if(generate_flips) {
            cv::flip(scratch, scratch2, 1);
            ret.labels.push_back(label);
            images.push_back(scratch2.clone());
            scratch2.release();
        }

        scratch_channel[0].release();
        scratch_channel[1].release();
        scratch_channel[2].release();
        scratch.release();

        sample_count++;
    }

    int data_idx = 0;
#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for(int i=0;i<num_samples;i++) {
        printf("Processing image #%d\n", i);
        data.row(i) = extract_features(images[i], mean, whitener, centroids);
    }

    ret.data = data.clone();
    return ret;
}

int main(int argc, char* argv[]) {
    if(argc<4) {
        cout << argv[0] << " <whitening.yaml> <centroids.yaml> <training.bin>" << endl;
        return 0;
    }

    cv::Mat mean, whitener;
    cv::Mat centroids;

    // Load the whitening data
    cv::FileStorage fs1(argv[1], cv::FileStorage::READ);
    fs1["mean"] >> mean;
    fs1["whitener"] >> whitener;
    fs1.release();

    // Load the centroids
    cv::FileStorage fs2(argv[2], cv::FileStorage::READ);
    fs2["centroids"] >> centroids;
    fs2.release();

    training_t tdata = load_data_neural(argv[3], true, mean, whitener, centroids);

    cv::FileStorage fs3("./features.yaml", cv::FileStorage::WRITE);
    fs3 << "data" << tdata.data;
    fs3 << "labels" << tdata.labels;
    fs3.release();
}
