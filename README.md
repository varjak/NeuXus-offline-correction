This is offline version of the NeuXus online EEG MR-artifact correction:
https://github.com/LaSEEB/NeuXus

My main contribution is the real-time algorithm present in file correct.py (in the online version, it is in: NeuXus/neuxus/nodes/correct.py).

The output should be virtually the same to the one from the online version.

It is specially useful for debugging, because it's written in simple python, allows breakpoints and plotting. Yet, it can also be used to correct data retrospectively (from files). 

It includes two files, one simulating the Gradient Artifact (GA) and another the Pulse Artifact (PA), each as a sinusoid.
