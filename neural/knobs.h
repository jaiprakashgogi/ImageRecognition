#define CENTROIDS 4000
#define PATCH_SIZE 6
#define IMG_SIZE 32
#define STANDARDIZE_EPSILON 10.0f

#define _USE_OPENMP

// Derived from the knobs above
#define NUM_BYTES_PER_IMG IMG_SIZE * IMG_SIZE * 3
#define SAMPLES (CENTROIDS*300)
