# 4803Project
How to use the code<br>
### DataSet.py
There are three different DataSets are NumsDataset, CharsDataset, FlowerDataset. 
DataSet(batch=64,dsize=64,loadfile=None)<br>
# Parameters:
batch: int, default=64<br>
The batch size of the dataset<br>
dsize: int, default=64<br>
The size of image<br>
loadfile: str,default=None<br>
Our dataset provide the save_dataset method to save the train_dataset,test_dataset as tf.data, and save val as np array
loadfile will be the name of file which want to load.<br>

# Attributes
X:ndarray of shape (n,img_shape)<br>
The image dataset without any batch.<br>
Y:ndarray of shape (n,)<br>
The label <br>
train_dataset: tf.Dataset<br>
The train dataset<br>
test_dataset: tf.Dataset<br>
The train dataset<br>
model: The neural network model<br>
x_val: image valid set<br>
y_val: label valid set<br>
dsize: The size of image<br>

# Methods
train(epoch): Training the dataset with current model, default epoch = 10<br>
test(): Testing the test dataset with current model, return [test_loss,test_accuracy] <br>
predict(X): predict X with current model, return predictions<br>
plot(index): plot the X[index] image<br>
display(index): display the X[index] image and label by predict<br>
save_model(name): save current model to Model file, by filename is name.h5<br>
load_model(name): load model from Model file which filename is name.h5<br>
save_dataset(name,dataset): save dataset to DataSet file. dataset is which dataset to save, if dataset == 'Both', we save both train and test dataset, otherwise if dataset == 'Train', we save the train dataset, and if dataset == 'Test', we save the test dataset.<br>
load_dataset(name,dataset): Loading dataset from Dataset file, dataset decide which tpye dataset to load.<br>
ld_data(name): loading the dataset from Dataset file, by filename is name, then use baseCNN as default model<br>
set_model(net): set the model for our dataset<br>
summary(): print our model<br>
svm(): use svm to predict our dataset<br>
RandomForest(n_estimators=n_estimators): use RandomForest to predict our dataset<br>
DecisionTree(): use DecisionTree to predict our dataset<br>
KNN(n_neighbors=n_neighbors): use KNN to predict our dataset<br>
LDA(n_estimators=n_estimators): use LDA to predict our dataset<br>
QDA(n_estimators=n_estimators): use QDA to predict our dataset<br>
show() plot the image with predicted label and expected label by choose model<br>
showNN() plot the image with predicted label and expected label by neural network<br>
classes(x) return the mapping of x, such as classes(10) = 'A' in chars dataset<br>
plot_accuary(NN = None): plot the bar to compare the accuary of 'LDA','QDA','KNN','RandomForest','svm' model

### model.py
This file includes the neural networks<br>
vgg13,vgg15,vgg19,baseCNN,AlexNet_c,AlexNet,DenseNet121 and MultLayer
