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

  random_shuffle(seeds.begin(), seeds.end());

  cv::Mat output;
  for (int cont = 0; cont < matrix.rows; cont++)
    output.push_back(matrix.row(seeds[cont]).clone());

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

void writeMatFile(const char *fileName, cv::Mat mat, const char* name) {
    FILE *file = fopen( fileName, "wb" );

    // If mat is ND we write multiple copies with _n suffixes
    int depth = mat.channels();
    for( int d=0; d<depth; d++) {

        // Assumption that we are always dealing with double precision
        uint32_t MOPT = 0000;
        fwrite(&MOPT, 1, sizeof( uint32_t), file);
        uint32_t    mrows = mat.rows;
        uint32_t    ncols = mat.cols;
        uint32_t    imagef = 0;

        char nameBuff[ strlen( name ) + 10];
        strcpy(nameBuff, name);
        if( depth>1) {
            char suffix[5];
            sprintf(suffix, "_L%d", d+1);
            strcat(nameBuff, suffix);
        }

        uint32_t    nameLength = strlen( nameBuff ) + 1;
        fwrite( &mrows, 1, sizeof( uint32_t), file);
        fwrite( &ncols, 1, sizeof( uint32_t), file);
        fwrite( &imagef, 1, sizeof( uint32_t), file);
        fwrite( &nameLength, 1, sizeof( uint32_t), file);
        fwrite( nameBuff, nameLength, 1, file );

        for( int col = 0; col<ncols; col++ ) {
            for( int row=0; row<mrows; row++ ) {
                cv::Scalar sc = mat.at<float>(row, col); //cvGet2D(mat, row, col);
                fwrite(&(sc.val[d]), 1, sizeof( double ), file);
            }
        }
    }
    fclose( file );
}
