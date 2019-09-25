# Neural Network Comparison - Cancer Identification
 Using some simple neural network strategies to compare performance when categorizing breast histopathology images for use in cancer identification. 
 
Note the purpose of this repo is not to create the best performing model, but to compare the results different NN layers has on the performance of the model. Note that work on this repo is ongoing.

*This is designed to run in Google Colab not in Jupyter, although this can be easily modified*
*Note that a kaggle account is required to run this code, as kaggle is where the data is downloaded from. In order for it to run, you must download your API JSON file from kaggle, and upload it into Colab at the indicated time*

### Data Visualization
We start off with getting the data into the notebook and visualizing the data on a basic level.

### Models
We decided to compare the following models:

- Perceptron
- Simple Dense
- Simple Dense with Dropout
- Dense with Residual Layers
- Simple CNN
- Deep CNN without Residual Layers
- Deep CNN with Residual Layers

Note that the models are all definted in the seperate python files in order to simplify the code within the notebook. We import these functions at the beginning. If you want more details on the particular architectures of the models, all of the source code is within the python files.

### Model Comparisons
We compare the different performances of the models. 
