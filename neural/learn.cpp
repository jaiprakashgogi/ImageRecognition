#include "everything.h"

using namespace std;

void load_data(cv::Mat samples, char* filename) {
    printf("Opening dataset: %s\n", filename);
    // Ensure the file can be loaded
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        printf("The file does not exist: %s\n", filename);
        return;
    }

    // Find the number of elements in this
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    if(size % (NUM_BYTES_PER_IMG+1) != 0) {
        printf("Non-integer number of elements in the given file. Exitting.\n");
        return;
    }

    const int num_samples = size / (NUM_BYTES_PER_IMG + 1);
    const int rcount = PATCH_SIZE * PATCH_SIZE * 3;
    const int convolved_size = (IMG_SIZE-PATCH_SIZE) * (IMG_SIZE-PATCH_SIZE);
    cv::Mat data = cv::Mat::zeros(rcount, convolved_size * num_samples, CV_32FC1);

    printf("Number of images in dataset = %d\n", num_samples);

    // Rewind to the beginning
    fseek(fp, 0, SEEK_SET);

    // Loop through the entire dataset
    int sample_count = 0;
    while(sample_count < num_samples) {
        uchar label;
        uchar *img_data = new uchar[NUM_BYTES_PER_IMG];
        if(fread(&label, sizeof(uchar), 1, fp) != 1) {
            printf("Unable to read label for image %d\n", sample_count);
            return;
        }
        if(fread(img_data, sizeof(uchar), NUM_BYTES_PER_IMG, fp) != NUM_BYTES_PER_IMG) {
            printf("Unable to read image data for image %d\n", sample_count);
            return;
        }
        sample_count += 1;

        if(sample_count%1000 == 0){
            printf("Sample count = %d\n", sample_count);
        }
    }
    fclose(fp);
    return;
}

int main(int argc, char* argv[]) {
    const int rcount = PATCH_SIZE * PATCH_SIZE * 3;
    cv::Mat data = cv::Mat(rcount, SAMPLES, CV_32FC1);
    cv::Mat centroids = cv::Mat(rcount, CENTROIDS, CV_32FC1);
    cv::Mat zca_u = cv::Mat(rcount, rcount, CV_32FC1);
    cv::Mat zca_m = cv::Mat(rcount, 1, CV_32FC1);

    cv::Mat test;
    load_data(test, "../data/data_batch_1.bin");
    return 0;
}
