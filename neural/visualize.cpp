#include "everything.h"

using namespace std;
cv::Mat visualize_patches(cv::Mat src, cv::Mat data) {
    const int num_rows = (IMG_SIZE-PATCH_SIZE + 1) * PATCH_SIZE;
    const int num_cols = IMG_SIZE + 5 + (IMG_SIZE-PATCH_SIZE + 1) * PATCH_SIZE;
    const int patch_area = PATCH_SIZE * PATCH_SIZE;
    cv::Mat visual = cv::Mat::zeros(num_rows, num_cols, CV_8UC3);

    cv::Mat aux = visual.colRange(0, IMG_SIZE).rowRange(0, IMG_SIZE);
    src.copyTo(aux);

    int i=0;
    for(int y=0;y<IMG_SIZE-PATCH_SIZE;y++) {
        for(int x=0;x<IMG_SIZE-PATCH_SIZE;x++) {
            int startx = IMG_SIZE + 5 + PATCH_SIZE * x;
            int endx = IMG_SIZE + 5 + PATCH_SIZE * (x+1);
            int starty = PATCH_SIZE * y;
            int endy = PATCH_SIZE * (y+1);
            aux = visual.rowRange(starty, endy).colRange(startx, endx);

            cv::Mat temp = data.row(i).clone().reshape(3, PATCH_SIZE);
            temp.copyTo(aux);
            i++;
        }
    }

    return visual;
}

cv::Mat visualize_patches_std(cv::Mat src, cv::Mat data) {
    const int num_rows = (IMG_SIZE-PATCH_SIZE + 1) * PATCH_SIZE;
    const int num_cols = IMG_SIZE + 5 + (IMG_SIZE-PATCH_SIZE + 1) * PATCH_SIZE;
    const int patch_area = PATCH_SIZE * PATCH_SIZE;
    cv::Mat visual = cv::Mat::zeros(num_rows, num_cols, CV_8UC3);

    cv::Mat aux = visual.colRange(0, IMG_SIZE).rowRange(0, IMG_SIZE);
    src.copyTo(aux);

    int i=0;
    for(int y=0;y<IMG_SIZE-PATCH_SIZE;y++) {
        for(int x=0;x<IMG_SIZE-PATCH_SIZE;x++) {
            int startx = IMG_SIZE + 5 + PATCH_SIZE * x;
            int endx = IMG_SIZE + 5 + PATCH_SIZE * (x+1);
            int starty = PATCH_SIZE * y;
            int endy = PATCH_SIZE * (y+1);
            aux = visual.rowRange(starty, endy).colRange(startx, endx);

            //cout << data.row(i) << endl;
            cv::Mat temp = data.row(i).clone().reshape(3, PATCH_SIZE);
            cv::Mat temp2;
            temp.convertTo(temp2, CV_8UC3, 255, 128);
            temp2.copyTo(aux);
            i++;
        }
    }

    return visual;
}

cv::Mat visualize_patches_zca(cv::Mat src, cv::Mat data) {
    const int num_rows = (IMG_SIZE-PATCH_SIZE + 1) * PATCH_SIZE;
    const int num_cols = IMG_SIZE + 5 + (IMG_SIZE-PATCH_SIZE + 1) * PATCH_SIZE;
    const int patch_area = PATCH_SIZE * PATCH_SIZE;
    cv::Mat visual = cv::Mat::zeros(num_rows, num_cols, CV_8UC3);

    cv::Mat aux = visual.colRange(0, IMG_SIZE).rowRange(0, IMG_SIZE);
    src.copyTo(aux);

    int i=0;
    for(int y=0;y<IMG_SIZE-PATCH_SIZE;y++) {
        for(int x=0;x<IMG_SIZE-PATCH_SIZE;x++) {
            int startx = IMG_SIZE + 5 + PATCH_SIZE * x;
            int endx = IMG_SIZE + 5 + PATCH_SIZE * (x+1);
            int starty = PATCH_SIZE * y;
            int endy = PATCH_SIZE * (y+1);
            aux = visual.rowRange(starty, endy).colRange(startx, endx);

            //cout << data.row(i) << endl;
            cv::Mat temp = data.row(i).clone().reshape(3, PATCH_SIZE);
            cv::Mat temp2;
            temp.convertTo(temp2, CV_8UC3, 255, 128);
            temp2.copyTo(aux);
            i++;
        }
    }

    return visual;
}

cv::Mat visualize_kmeans_centroids(cv::Mat centroids) {
    const int num_centroids = centroids.rows;
    const int num_per_row = 80;
    const int num_rows = num_centroids / num_per_row;

    cv::Mat visual = cv::Mat::zeros((PATCH_SIZE+1)*num_rows, (PATCH_SIZE+1)*num_per_row, CV_8UC3);

    int i=0;
    for(int y=0;y<num_rows;y++) {
        for(int x=0;x<num_per_row;x++) {
            if(i>=num_centroids) {
                break;
            }
            cv::Mat temp = centroids.row(i).clone().reshape(3, PATCH_SIZE);
            cv::Mat patch;
            temp.convertTo(patch, CV_8UC3, 255, 128);

            int startx = x*(PATCH_SIZE+1);
            int endx = (x+1)*(PATCH_SIZE+1);
            int starty = y*(PATCH_SIZE+1);
            int endy = (y+1)*(PATCH_SIZE+1);

            cv::Rect r = cv::Rect(startx, starty, PATCH_SIZE, PATCH_SIZE);
            cv::Mat aux = visual(r);
            patch.copyTo(aux);

            i++;
        }
        if(i>=num_centroids) {
            break;
        }
    }

    return visual;
}
