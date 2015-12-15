# ImageRecognition
Machine Learning Fall 2015 project

## Neural networks
An implementation of a neural network from scratch to classify the CIFAR-10 dataset.

### How to execute
```
%> make learn       # Generates executables for learning the neural network
%> make classify    # Build executables for classifying a given image
%> make tools       # Build Supporting tools

%> ./learn_centroids <files>    # Learns centroids of patches
# Produces whitening.yaml and centroids.yaml

%> ./learn_network whitening.yaml centroids.yaml
# produces network.yaml

%> ./classify network.yaml <images>
# Prints out the result
```

Installing `mkoctfile`:
```
sudo apt-get install liboctave-dev
```
