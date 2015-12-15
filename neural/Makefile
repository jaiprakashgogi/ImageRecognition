EXTRAS = utils.cpp visualize.cpp preprocess.cpp
LIBRARIES = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_calib3d -lopencv_videostab -lopencv_objdetect -lopencv_video 
FLAGS = -std=c++0x -ggdb -O3 -Ofast -fopenmp -Wno-write-strings

learn:
	g++ neural.cpp mlp.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -o learn_neural

centroids:
	g++ learn.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -o learn_centroids

tools:
	g++ visualize_centroids.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -o v_centroids

yaml:
	g++ yaml_to_mat.cpp $(EXTRAS) $(LIBRARIES) $(FLAGS) -I/work/school/10-601/project/references/cvmatio/include -lcvmatio -fpermissive -o yaml2mat 

train_centroids:
	./learn_centroids ../data/small_data_batch_all.bin > ../../logs/run_centroids.log

train:
	./learn_neural whitening.yaml centroids.yaml ../data/small_data_batch_a_train.bin #> ../../logs/run_neural.log

scrub:
	cat features.yaml | sed "s/0\.,//g" | sed '/^\s*$$/d' > test
