#include "everything.h"    


int parseLine(char* line) {
    int i = strlen(line);
    while (*line < '0' || *line > '9')
        line++;

    line[i-3] = '\0';

    i = atoi(line);
    return i;
}

int getMemValue() { //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];


    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmSize:", 7) == 0) {
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

cv::Mat shuffle_rows(const cv::Mat &matrix) {
  std::vector <int> seeds;
  for (int cont = 0; cont < matrix.rows; cont++)
    seeds.push_back(cont);

  cv::randShuffle(seeds);

  cv::Mat output;
  for (int cont = 0; cont < matrix.rows; cont++)
    output.push_back(matrix.row(seeds[cont]));

  return output;
}

bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

cv::Mat feature_mean(cv::Mat data) {
    const int num_vectors = data.rows;
    const int feature_size = data.cols;

    cv::Mat ret = cv::Mat::zeros(1, feature_size, CV_32FC1);
    for(int i=0;i<num_vectors;i++) {
        cv::Mat t;
        data.row(i).convertTo(t, CV_32FC1);
        ret = ret + t;
    }
    ret /= num_vectors;

    return ret;
}
