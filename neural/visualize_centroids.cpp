#include "everything.h"

using namespace std;

int main(int argc, char* argv[]) {
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    cv::Mat centroids;
    fs["centroids"] >> centroids;

    cv::Mat visual = visualize_kmeans_centroids(centroids);
    cv::imshow("Centroids", visual);
    cv::waitKey(0);
    return 0;
}
