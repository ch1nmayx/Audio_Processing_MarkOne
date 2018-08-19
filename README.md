#### Audio Signal Processing

@Chinmay Sinha
@UCD
@Final Research Project 

In order to tackle theMultiple Genre Problem, we will use the Custom LSTMModel , which performs best on our training dataset and train it on completedataset of 1000 songs and will record the audio genre of a test song at every timeinterval. The time interval will be decided based on the window size of the fastFourier transformation function used in the algorithm to convert audio files intospectrogram. Generally, we have usedWINDOWSIZE=ƦƤƨƬ, which gives usthe change in everyƤ:Ƨseconds, However, if we will decrease the window size itwill create more frequently changing spectrogram which will be useful in tacklingquick changing songs, but we would need a more computing power and moreparameters to train such dataset, and increasing the window size would decreaseour frequency to monitor change in the song, hence, we would go ahead with thewindow size of 2048.

Next, we will take the output of the an audio file and get the output from themodel as a probability of genre for every time unit of the window size used, in ourcase we will get an output of genre with a probability ofƤ:Ƨƫseconds. The outputwe will get will be in JSON format, and we would parse the JSON file into apandas dataframe using the parser function given in Code Listings as6. Next,when we have the dataset parsed we will feed the parsed pandas dataframe into aJavascript application and create a visualization which will give the genre of thesong as the song proceeds. For this visualization, we will use the song by theartistDJ Shadow, which can be classified as Classical in the start and hip-hop inthe rest of the song.

he Plotly link to the plot generated is https://plot.ly/~chinmaysinha/5/ . We can conclude thatclassification of a song is a subjective affair and is highly biased on the listener andit natural for a song to lie in different genres at different parts of the playback.Hence, instead of getting one specific genres, we can get genres for multiple timeperiods. We were able to achieveƪƬ%accuracy on a single genre classifier basedon our methods, we were also able to test various ImageNet algorithms ourspectrogram dataset, Some classifiers like Resnet50x, VGG19 and AlexNet gavedecent results as compared to the Custom LSTM which was designed keepingothers as base. However, real time recommendation of song can be of use invarious applications like Spotify and iTunes. Javascript applications like ours canalso be used to increase the confidence of prediction in Shazam like applications.

We will start with installtion of Anaconda, Keras and Theano on our system.DependenciesHere’s a summary list of the tools and libraries we use for deeplearning on Windows 10 (Version 1709 OS Build 16299.371):Visual Studio 2015 Community Edition which is used for its C/C++compiler (not its IDE) and SDK. This specific version has been selecteddue to Windows Compiler Support in CUDA.Anaconda (64-bit) w. Python 3.6 (Anaconda3-5.2.0).Python 2.7 (Anaconda2-5.2.0) [no Tensorflow support] with MKL2018.0.3 with NumPy, SciPy, and other scientific libraries.cuDNN v7.0.4 (Nov 13, 2017) for CUDA 9.0.176 which is used to runvastly faster convolution neural networks.Keras 2.1.6 with three different backends: Tensorflow-gpu 1.8.0,CNTK-gpu 2.5.1, and MXNet-cuda90 1.2.0.Theano is a legacy backend no longer in active developmentThen we install Keras, tensorflow-gpu, cntk-gpu, mxnet-cu90 and pytorch0.4.0 as follows





