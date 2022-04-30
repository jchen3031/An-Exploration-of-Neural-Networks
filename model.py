# The  
import tensorflow as tf
from tensorflow.keras import layers
def vgg13(input_shape, classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, 1, input_shape=input_shape,padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(64, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(128, 3, 1, padding= 'same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(128, 3, 1, padding= 'same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(256, 3, 1, padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(256, 3, 1, padding='same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding= 'same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation= 'softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def vgg15(input_shape, classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, 1, input_shape=input_shape,padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(64, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(128, 3, 1, padding= 'same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(128, 3, 1, padding= 'same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(256, 3, 1, padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(256, 3, 1, padding='same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding= 'same',activation= 'relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding = 'same'))
    
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding= 'same',activation= 'relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation= 'softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def vgg19(input_shape, classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, 1, input_shape=input_shape,padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(64, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(128, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(128, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(256, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(256, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(256, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    model.add(layers.Conv2D(256, 3, 1, padding='same',activation='relu',kernel_initializer= 'uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding = 'same'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512, 3, 1, padding='same',activation= 'relu'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation= 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def baseCNN(input_shape, classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, 3, 1, input_shape=input_shape,padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding = 'same'))
    model.add(layers.Conv2D(32, 3, 1,padding= 'same',activation= 'relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding = 'same'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation= 'relu'))
    model.add(layers.Dense(32, activation= 'relu'))
    model.add(layers.Dense(classes, activation= 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def AlexNet_c(input_shape, classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, 11, strides= 4, input_shape=input_shape,padding= 'same',activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding = 'same'))
    model.add(layers.Conv2D(256, 5, strides = 1 ,padding= 'same',activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3),padding = 'same'))
    model.add(layers.Conv2D(256, 5, strides = 1 ,padding= 'same',activation= 'relu'))
    model.add(layers.Conv2D(256, 1, strides = 1 ,padding= 'same',activation= 'relu'))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding = 'same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation= 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def AlexNet(input_shape, classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, 11, strides= 4, input_shape=input_shape,padding= 'same',activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3),padding = 'same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3),padding = 'same'))
    model.add(layers.Conv2D(192, 5, 1 ,padding= 'same',activation= 'relu'))
    model.add(layers.Conv2D(192, 5, 1 ,padding= 'same',activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3),padding = 'same'))
    model.add(layers.Conv2D(192, 5, 1 ,padding= 'same',activation= 'relu'))
    model.add(layers.Conv2D(192, 5, 1 ,padding= 'same',activation= 'relu'))
    
    model.add(layers.MaxPooling2D(pool_size=(3, 3),padding = 'same'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation= 'relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(512, activation= 'relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(classes, activation= 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def DenseNet121(input_shape,classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.applications.densenet.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    ))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model
def MultLayer(input_shape, classes):
    model = tf.keras.Sequential()
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation= 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation= 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return model