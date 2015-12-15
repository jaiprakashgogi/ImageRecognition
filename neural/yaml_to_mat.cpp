#include "everything.h"

using namespace std;

int main(int argc, char* argv[]) {
    if(argc<4) {
        cout << argv[0] << " <src.yaml> <matrix-name> <output.ybin>" << endl;
    }

    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    cv::Mat src;
    fs[argv[2]] >> src;
    fs.release();

    cv::Mat test = src.row(0).reshape(3, 6);
    vector<cv::Mat> split_up;
    cv::split(test, split_up);

    cout << split_up[0] << endl;
    cout << split_up[1] << endl;
    cout << split_up[2] << endl;

    writeMatFile(argv[3], src, "centroids");

    return 0;
}
