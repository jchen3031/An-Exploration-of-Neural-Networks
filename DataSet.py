import datetime
import tensorflow as tf
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import model
import datetime
import sys
import time
import matplotlib.pyplot as plt
import io
import gzip
from PIL import Image
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class DataSet(object):
    def __init__(self,batch = 64,dsize = 64,loadfile = None):
        self.X = None
        self.Y = None
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.x_val = None
        self.y_val = None
        self.dsize = dsize
        if loadfile != None:
            print('loading')
            start = time.time()
            self.ld_data(loadfile)
            end = time.time()
            print(end - start)
        else:
            start = time.time()
            self.data(batch, dsize)
            end = time.time()
            print(end - start)
    def train(self,epoch = 10):
        CNN = self.model
        log_dir="logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
        tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stop=tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', min_delta=0.002, patience=10, mode='auto')
        callbacks = [tensorboard_callback, early_stop]
        return CNN.fit(self.train_dataset, epochs=epoch,verbose=1, validation_data = (self.x_val,self.y_val), callbacks=callbacks)
    def test(self):
        return self.model.evaluate(self.test_dataset)
    def predict(self, X):
        predictions = self.model.predict(X)
        pred = np.argmax(predictions, axis=1)
        return pred
    def display(self,index):
        img = self.X[index]
        plt.imshow(img)
        img = np.expand_dims(img,0)
        pred= np.argmax(self.model.predict(img), axis=1)
        print(pred[0])
    def plot(self,index):
        img = self.X[index]
        plt.imshow(img)
    def save_model(self, name = 'model'):
        path = 'Model/'+name+'.h5'
        self.model.save(path)
        print(f'the model has been saved in {path}')
    def load_model(self, name = 'model'):
        path = 'Model/'+name+'.h5'
        print(f'the model has been loaded from {path}')
        self.model = tf.keras.models.load_model(path)
    def save_dataset(self, name = 'data',dataset = 'Train'):
        start = time.time()
        print('start saving data')
        dirpath = 'DataSet/'
        if dataset == 'Both':
            path1 = dirpath+name+'/'+name+'-'+'Train'
            path2 = dirpath+name+'/'+name+'-'+'Test'
            tf.data.experimental.save(self.train_dataset, path1)
            tf.data.experimental.save(self.test_dataset, path2)    
        elif dataset == 'Train':
            path = dirpath+name+'/'+name+'-'+dataset
            tf.data.experimental.save(self.train_dataset, path)
        elif dataset == 'Test':
            path = dirpath+name+'/'+name+'-'+dataset
            tf.data.experimental.save(self.test_dataset, path)
        path1 = dirpath+name+'/'+name+'-'+'valx'
        path2 = dirpath+name+'/'+name+'-'+'valy'
        np.save(path1,self.x_val)
        np.save(path2,self.y_val)
        print(f'taking {time.time()-start} seconds')
    def load_dataset(self,name = 'data',dataset = 'Train'):
        dirpath = 'DataSet/'
        if dataset == 'Both':
            path1 = dirpath+name+'/'+name+'-'+'Train'
            path2 = dirpath+name+'/'+name+'-'+'Test'
            self.train_dataset = tf.data.experimental.load(path1)
            self.test_dataset = tf.data.experimental.load(path2)
        elif dataset == 'Train':
            path = dirpath+name+'-'+dataset
            self.train_dataset = tf.data.experimental.load(path)
        elif dataset == 'Test':
            path = dirpath+name+'-'+dataset
            self.test_dataset = tf.data.experimental.load(path)
        path1 = dirpath+name+'/'+name+'-'+'valx.npy'
        path2 = dirpath+name+'/'+name+'-'+'valy.npy'
        self.x_val = np.load(path1)
        self.y_val = np.load(path2)
    def ld_data(self, name):
        self.load_dataset(name, 'Both')
        self.model = model.CNNmodel(input_shape=(28,28,1), classes=10)
        print(f'default set the model as CNN')
    def set_model(self,net = None):
        classes = self.y_val[0].shape[0]
        input_size = self.x_val[0].shape
        print(type(net))
        if len(input_size) == 2:
            input_size = (input_size[0],input_size[1],1)
        print(input_size,classes)
        if net == 'Alex':
            self.model = model.AlexNet(input_size,classes)
        elif net == 'vgg19':
            self.model = model.vgg19(input_size,classes)
        elif net == 'vgg13':
            self.model = model.vgg19(input_size,classes)
        else:
            self.model = net(input_size,classes)
        print(self.model)
    def summary(self):
        self.model.summary()
    def show_train_dataset(self):
        ls = list(self.train_dataset.as_numpy_iterator())
        return ls
    def back(self,y):
        r = []
        for i in y:
            ls = [j for j in range(len(i)) if i[j] == 1]
            r+=ls
        r = np.array(r)
        return r
    def svm(self, c = 10, dsize = 28):
        model = svm.SVC(C= c)
        return self.fit_predict(model, dsize)
    def RandomForest(self,n_estimators = 25,dsize = 28):
        model = RandomForestClassifier(n_estimators=n_estimators, max_features = 28)
        return self.fit_predict(model, dsize)
    def DecisionTree(self,dsize = 28): # fastest, but accuacy is low
        model = DecisionTreeClassifier(max_features = 28)
        return self.fit_predict(model, dsize)
    def KNN(self,n_neighbors,dsize = 28):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return self.fit_predict(model, dsize)
    def LDA(self,dsize = 28):
        model = LinearDiscriminantAnalysis()
        return self.fit_predict(model, dsize)
    def QDA(self,dsize = 28):
        model = QuadraticDiscriminantAnalysis()
        return self.fit_predict(model, dsize)
    def fit_predict(self, model = None, dsize = 28, show = False):
        print(f'using model {model}')
        x_train,y_train,x_test,y_test = self.get_data(dsize)
        start = time.time()
        model.fit(x_train,y_train)
        end = time.time()
        print(f'Takes {end - start} seconds for fitting the model')
        pred = model.predict(x_test)
        result = np.where(y_test-pred ==0,1,0)
        accuacy = result.sum()/len(result)
        self.predict = pred
        if show:
            self.show(model,x_test,y_test)
        return accuacy
    def show(self,model = None,X = None,Y = None):
        if model == None:
            model = self.model
        if X == None:
            X = self.X
            Y = self.Y
        plt.figure(figsize=(7, 7))
        print(X[0].shape)
        for i, (image, label) in enumerate(zip(X[:15],Y[:15])):
            prediction = model.predict([x[i]])
            plt.subplot(3, 5, i + 1)
            plt.axis('off')
            image = image.reshape((28,28))
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('predicted %s' % self.classes(prediction[0]) + '\nexpected %s' % self.classes(label))
        plt.show()
    def showNN(self):
        ls = list(self.test_dataset.as_numpy_iterator())[0]
        X,Y = ls
        Y = self.back(Y)
        predictions = self.model.predict(X)
        prediction = np.argmax(predictions, axis=1)
        plt.figure(figsize=(7, 7))
        img = X[0]
        print(np.amax(img),np.amin(img))
        for i, (image, label) in enumerate(zip(X[:15],Y[:15])):
            plt.subplot(3, 5, i + 1)
            plt.axis('off')
            plt.imshow(np.clip(image, 0, 1))
            plt.title('predicted %s' % self.classes(prediction[i]) + '\nexpected %s' % self.classes(label))
        plt.show()
    def get_data(self,dsize):
        ls = list(self.train_dataset.as_numpy_iterator())
        n = len(ls)
        x_train_ls = []
        y_train_ls = []
        for i,j in enumerate(ls):
            x_train,y_train = j
            if (i+1)%2 == 0 or i+1== n:
                print('\r',f'{i+1}/{n} loading the data',end = '')
            for x in x_train:
                if len(x.shape) == 3:
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = cv2.resize(x,(dsize,dsize))
                x = x.reshape((1,-1))
                x_train_ls.append(x)
            y_train = self.back(y_train)
            y_train_ls.append(y_train)
        x_train = np.concatenate(x_train_ls)
        y_train = np.concatenate(y_train_ls)
        print()
        ls = self.test_dataset.as_numpy_iterator()
        x_ls = []
        y_ls = []
        for x,y in ls:
            for img in x:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(dsize,dsize))
                img = img.reshape((1,-1))
                x_ls.append(img)
            y = self.back(y)
            y_ls.append(y)
        x_test = np.concatenate(x_ls)
        y_test = np.concatenate(y_ls)
        return x_train,y_train,x_test,y_test
    def classes(self,x):
        return str(x)
    def plot_accuary(self):
        accuary = []
        label = ['LDA','QDA','KNN','RandomForest','svm']
        accuary.append(self.LDA())
        accuary.append(self.QDA())
        accuary.append(self.KNN(n_neighbors = 10))
        accuary.append(self.RandomForest(n_estimators = 100))
        accuary.append(self.svm(c = 10))
        accuary = np.array(accuary)*100
        plt.bar(label, accuary)
        for a,b in zip(label,accuary): 
            plt.text(a, b+0.05, '%.2f ' % b, ha='center', va= 'bottom',fontsize=11) 
class NumsDataset(DataSet):
    def __init__(self,batch = 64,dsize = 28, loadfile = None):
        super(NumsDataset,self).__init__(batch,dsize,loadfile)
    def data(self,batch, dsize):
        f = gzip.open('train-images-idx3-ubyte.gz','r')
        image_size = dsize
        num_images = 60000
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        x_train = data.reshape(num_images, image_size, image_size)
        
        f = gzip.open('t10k-images-idx3-ubyte.gz','r')
        num_images = 10000
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        x_test = data.reshape(num_images, image_size, image_size)
        print(x_train.shape,x_test.shape)
        
        f = gzip.open('train-labels-idx1-ubyte.gz','r')
        f.read(8)
        y_train = []
        for i in range(0,60000):   
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            y_train.append(labels[0])
        y_train = np.array(y_train)
        
        f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
        f.read(8)
        y_test = []
        for i in range(0,10000):   
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            y_test.append(labels[0])
        #print(y_train)
        y_test = np.array(y_test)
        print(y_train.shape,y_test.shape)
        input_shape = np.expand_dims(x_train,-1)
        input_shape = input_shape[0].shape
        self.input_shape = input_shape
        self.model = model.baseCNN(input_shape=self.input_shape, classes=10)
        
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch)
        self.test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch)
        self.x_val = x_test[-1000:]
        self.y_val = y_test[-1000:]
        self.X = x_train
        self.Y = y_train
class CharsDataset(DataSet):
    def __init__(self, batch = 64,dsize = 64,loadfile = None):
        super(CharsDataset,self).__init__(batch,dsize,loadfile)
    def data(self,batch,dsize = 64):
        files = glob.glob('./EnglishImg/EnglishImg/English/Img/*/Bmp/*/*.png')
        masks = glob.glob('./EnglishImg/EnglishImg/English/Img/*/Msk/*/*.png')
        n = len(files)
        X = []
        Y = []
        for i,f in enumerate(files):
            img = cv2.imread(f)
            mask = cv2.imread(masks[i])
            img = cv2.add(img,~mask)
            #if len(img.shape) == 3:
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(dsize,dsize),interpolation = cv2.INTER_CUBIC)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            Y.append(int(f.split('\\')[-2].split('e')[1])-1)
            X.append(img)
            if (i+1)%1000 == 0 or i+1==n:
                print('\r',f'{i+1}/{n} loading the files {f}',end='')
        X = np.stack(X)
        Y = np.array(Y)
        print('\n',X.shape,Y.shape)
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
        print(x_train.shape,y_train.shape)
        print(x_test.shape,y_test.shape)
        y_train = tf.keras.utils.to_categorical(y_train, 62)
        y_test = tf.keras.utils.to_categorical(y_test, 62)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch)
        
        self.input_shape = (dsize,dsize,3)
        self.model = model.AlexNet_c(input_shape=self.input_shape, classes=62)
        self.x_val = x_test[-1000:]
        self.y_val = y_test[-1000:]
        self.X = X
        self.Y = Y
    def display(self,index):
        img = self.X[index]
        plt.imshow(img)
        img = np.expand_dims(img,0)
        pred= np.argmax(self.model.predict(img), axis=1)
        print(pred[0])
        print(self.classes(pred[0]))
    def classes(self,x):
        if 0<=x and x<=9:
            return x
        if 10<=x and x<36:
            return chr(x+55)
        else:
            return chr(x+61)
class FlowerDataset(DataSet):
    def __init__(self,batch = 64,dsize = 64,loadfile = None):#/drive/MyDrive
        super(FlowerDataset,self).__init__(batch,loadfile)
    def _parse_image_function(self,example):
        train_feature_description = {
            'class': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(example, train_feature_description)
    def data(self,batch):
        self.train_files = glob.glob('flowerdataset/tfrecords-jpeg-192x192/train/*.tfrec')
        self.val_files = glob.glob('flowerdataset/tfrecords-jpeg-192x192/val/*.tfrec')
        self.test_files = glob.glob('flowerdataset/tfrecords-jpeg-192x192/test/*.tfrec')
        train_class = []
        train_images = []
        n = len(self.train_files)
        for j,i in enumerate(self.train_files):
            print('\r',f'{j+1}/{n} loading the training files {i}',end='')
            train_image_dataset = tf.data.TFRecordDataset(i)
            train_image_dataset = train_image_dataset.map(self._parse_image_function)
            classes = [int(class_features['class']) for class_features in train_image_dataset]
            train_class += classes
            images = [self.numpy(image_features['image'].numpy()) for image_features in train_image_dataset]
            train_images.append(np.stack(images))
            
        train_images = np.concatenate(train_images)
        train_class = np.array(train_class)
        self.X = train_images
        self.Y = train_class
        
        val_classes = []
        val_img = []
        for j,i in enumerate(self.val_files):
            print('\r',f'{j+1}/{n} loading the val files {i}',end='')
            train_image_dataset = tf.data.TFRecordDataset(i)
            train_image_dataset = train_image_dataset.map(self._parse_image_function)
            classes = [int(class_features['class']) for class_features in train_image_dataset]
            val_classes += classes
            images = [self.numpy(image_features['image'].numpy(dsize=dsize)) for image_features in train_image_dataset]
            val_img.append(np.stack(images))
        val_img = np.concatenate(val_img)
        val_classes = np.array(val_classes)
        self.x_val = val_img
        self.y_val = tf.keras.utils.to_categorical(val_classes, 104)
        self.x_train = train_images
        self.y_train = tf.keras.utils.to_categorical(train_class, 104)
        print(f'\n{train_images.shape}')
        
        self.input_shape = self.x_train[0].shape
        if len(self.input_shape) == 2:
            self.input_shape = (self.input_shape[0],self.input_shape[1],1)
        self.model = model.CustNN(input_shape=self.input_shape, classes=104)
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch)
        self.x_test = val_img
        self.y_test = self.y_val
        test_dataset = tf.data.Dataset.from_tensor_slices((val_img, self.y_val))
        self.test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch)
        
    def numpy(self, img, dsize = 64):
        img = np.array(Image.open(io.BytesIO(img)))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize = (dsize,dsize),interpolation = cv2.INTER_CUBIC)
        return img.astype('uint8')
    def classes(self,x):
        CLASSES = ['pink primrose','hard-leaved pocket orchid', 'canterbury bells', 'sweet pea','wild geranium','tiger lily','moon orchid','bird of paradise', 'monkshood','globe thistle',# 00 - 09
'snapdragon',"colt's foot",'king protea','spear thistle', 'yellow iris','globe-flower','purple coneflower','peruvian lily',    'balloon flower','giant white arum lily', # 10 - 19
'fire lily','pincushion flower','fritillary','red ginger','grape hyacinth','corn poppy','prince of wales feathers', 'stemless gentian', 'artichoke','sweet william',# 20 - 29
'carnation','garden phlox','love in the mist', 'cosmos','alpine sea holly',  'ruby-lipped cattleya', 'cape flower','great masterwort', 'siam tulip','lenten rose',# 30 - 39
'barberton daisy',  'daffodil','sword lily','poinsettia','bolero deep blue','wallflower','marigold','buttercup','daisy','common dandelion',# 40 - 49
'petunia','wild pansy','primula','sunflower','lilac hibiscus','bishop of llandaff','gaura','geranium','orange dahlia','pink-yellow dahlia',# 50 - 59
'cautleya spicata', 'japanese anemone','black-eyed susan', 'silverbush','californian poppy', 'osteospermum','spring crocus','iris','windflower','tree poppy',# 60 - 69
'gazania','azalea','water lily','rose','thorn apple','morning glory','passion flower','lotus','toad lily','anthurium',# 70 - 79
'frangipani','clematis','hibiscus','columbine','desert-rose','tree mallow','magnolia','cyclamen ','watercress','canna lily',# 80 - 89
'hippeastrum ','bee balm','pink quill','foxglove','bougainvillea','camellia','mallow','mexican petunia',  'bromelia','blanket flower',# 90 - 99
'trumpet creeper','blackberry lily','common tulip','wild rose']   # 100 - 102
        return CLASSES[x]