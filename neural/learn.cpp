#include "everything.h"

using namespace std;

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

    const int num_samples = size / (NUM_BYTES_PER_IMG + 1);
    //const int num_samples = 100;
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
        images.push_back(scratch.clone());

        // Calculate the individual patches
        cv::Mat pat = extract_patches(scratch, PATCH_SIZE);
        pat.clone().copyTo(data.rowRange(data_idx, data_idx+pat.rows));
        data_idx += pat.rows;

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
    if(argc<2) {
        cout << argv[0] << " <training.bin>" << endl;
        return 0;
    }

    const int rcount = PATCH_SIZE * PATCH_SIZE * 3;
    cv::Mat data = cv::Mat(rcount, SAMPLES, CV_32FC1);
    //cv::Mat zca_u = cv::Mat(rcount, rcount, CV_32FC1);
    //cv::Mat zca_m = cv::Mat(rcount, 1, CV_32FC1);
    vector<cv::Mat> images;

    // Reading this from a file takes longer than executing this.
    printf("Memory usage initially: %d KB\n", getMemValue());
    cv::Mat patches = load_data(argv[1], images);
    printf("Number of images loaded = %ld\n", (long)images.size());
    printf("Memory usage after loading images: %d KB\n", getMemValue());

    cv::Mat patches_std = normalize(patches, STANDARDIZE_EPSILON);
    printf("Memory usage: %d KB\n", getMemValue());

    //cv::Mat visual = visualize_patches(images[0], patches.rowRange(0, 676));
    //cv::Mat visual_std = visualize_patches_std(images[0], patches_std.rowRange(0, 676));
    //cv::imshow("Visualizing patches", visual);
    //cv::imshow("Visualizing patches - standardized", visual_std);
    //cv::waitKey(0);

    printf("Starting calculating the covariance matrix\n");
    //zca_learn(&zca_m, 0, &zca_u, patches, 0.1f);
    //cout << zca_m << endl;
    //printf("Size of zca_u = %d*%d\n", zca_u.rows, zca_u.cols);

    cv::Mat mean, whitener;
    mean = cv::Mat::zeros(1, PATCH_SIZE * PATCH_SIZE * 3, CV_32FC1);
    whitener = cv::Mat::zeros(PATCH_SIZE*PATCH_SIZE*3, PATCH_SIZE*PATCH_SIZE*3, CV_32FC1);
    cv::Mat patches_whitened = zca_white(patches_std, mean, whitener);
    printf("Memory usage after zca learning: %d KB\n", getMemValue());

    cv::FileStorage fs_whitener("./whitener.yaml", cv::FileStorage::WRITE);
    fs_whitener << "mean" << mean;
    fs_whitener << "whitener" << whitener;
    fs_whitener.release();

    /*for(int i=0;i<images.size();i++) {
        cv::Mat visual_normalized = visualize_patches_std(images[i], patches_std.rowRange(676*i, (i+1)*676));
        cv::Mat visual_whitened = visualize_patches_zca(images[i], patches_whitened.rowRange(676*i, (i+1)*676));
        cv::imshow("Visualizing patches - normalized", visual_normalized);
        cv::imshow("Visualizing patches - whitened", visual_whitened);
        cv::waitKey(0);
    }*/

    cout << "Starting kmeans on the whitened data" << endl;
    DECLARE_TIMING(kmeans_timer);
    cv::Mat labels;
    cv::Mat centroids;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.01);
    START_TIMING(kmeans_timer);
    cv::kmeans(patches_whitened, CENTROIDS, labels, termcrit, KMEANS_TRIALS, cv::KMEANS_PP_CENTERS, centroids);
    STOP_TIMING(kmeans_timer);
    cout << "Done with kmeans" << endl;
    cout << "It took " << GET_TIMING(kmeans_timer) << " ms to solve the centroids" << endl;

    cv::FileStorage fs("./centroids.yaml", cv::FileStorage::WRITE);
    fs << "centroids" << centroids;
    fs.release();

    printf("Number of rows = %d\n", patches.rows);
    printf("Memory usage after kmeans: %d KB\n", getMemValue());
    return 0;
}
