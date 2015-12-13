all:
	g++ learn.cpp utils.cpp visualize.cpp -std=c++0x -ggdb -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_calib3d -lopencv_videostab -lopencv_objdetect -lopencv_video -O3 -Ofast -Wno-write-strings -fopenmp
	g++ visualize_centroids.cpp utils.cpp visualize.cpp -std=c++0x -ggdb -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_calib3d -lopencv_videostab -lopencv_objdetect -lopencv_video -O3 -Ofast -Wno-write-strings -fopenmp -o v_centroids

