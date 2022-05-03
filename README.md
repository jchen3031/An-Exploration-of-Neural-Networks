# 4803Project
How to use the code<br>
DataSet.py <br>
There are three different DataSets are NumsDataset, CharsDataset, FlowerDataset. 
DataSet(batch=64,dsize=64,loadfile=None)<br>
Parameters:<br>
batch: int, default='64'<br>
The batch size of the dataset<br>
dsize: int, default='64'<br>
The size of image<br>
loadfile: str,default=None<br>
Our dataset provide the save_dataset method to save the train_dataset,test_dataset as tf.data, and save val as np array
loadfile will be the name of file which want to load.



