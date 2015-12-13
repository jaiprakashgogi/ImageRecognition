EXTRAS = utils.cpp visualize.cpp preprocess.cpp
LIBRARIES = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_calib3d -lopencv_videostab -lopencv_objdetect -lopencv_video 
FLAGS = -std=c++0x -ggdb -O3 -Ofast -fopenmp -Wno-write-strings

learn:
	g++ learn_neural.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -o learn_neural

centroids:
	g++ learn.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -o learn_centroids

tools:
	g++ visualize_centroids.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -o v_centroids

