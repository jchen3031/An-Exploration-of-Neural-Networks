# 4803Project
How to use the code<br>
DataSet.py <br>
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
