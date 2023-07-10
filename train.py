from keras.models import Model
from keras.layers import LSTM, Dense, Input
import tensorflow as tf
from functions import *
import pandas as pd


data = pd.read_csv("smp2020dataset/train/usual_train.csv").astype(str)
data.drop('数据编号', axis=1, inplace=True)

word, tfidf_matrix = clean_and_plot(data['文本'], ' ', False)

from sklearn import preprocessing

enc=preprocessing.LabelEncoder()
enc=enc.fit(['angry', 'happy', 'neutral', 'surprise', 'sad', 'fear'])
label = enc.transform(data['情绪标签'])

x_train = tfidf_matrix
Y_train = label

def mylstm():
    inputs = Input(shape=(57961))
    encoded = tf.reshape(inputs, [-1, 1, 57961])
    encoded = LSTM(2048, activation='tanh', return_sequences=True)(encoded)
    encoded = LSTM(1024, activation='tanh', return_sequences=True)(encoded)
    encoded = LSTM(512, activation='tanh', return_sequences=True)(encoded)
    encoded = LSTM(256, activation='tanh', return_sequences=True)(encoded)
    encoded = LSTM(128, activation='tanh', return_sequences=True)(encoded)
    encoded = LSTM(64, activation='tanh', return_sequences=True)(encoded)
    encoded_output = LSTM(10, activation='tanh')(encoded)
    encoded_output = Dense(6, activation="softmax")(encoded_output)
    model = Model(inputs=[inputs], outputs=encoded_output)
    return model


checkpoint_save_root = "./checkpoint"
checkpoint_save_path = checkpoint_save_root+'mylstm.h5'

model = mylstm()

cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_save_path,
                                                            monitor = 'val_accuracy',
                                                            save_best_only=True
                                                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                                monitor = 'val_accuracy'
                                ,factor = 0.5
                                ,patience=8
                            )
                ]
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(
                                                    learning_rate=0.001
                                                ),
                loss=
                # tf.keras.losses.BinaryCrossentropy(),
                tf.keras.losses.SparseCategoricalCrossentropy(),
                # tf.keras.losses.CategoricalCrossentropy(),

                metrics=[
                    # 'binary_accuracy'
                    # 'accuracy'
                    # tf.keras.metrics.CategoricalAccuracy()
                    # km.categorical_accuracy()
                    tf.keras.metrics.SparseCategoricalAccuracy()
                    # km.sparse_categorical_recall(),
                    # km.sparse_categorical_f1_score()
                    # tf.keras.metrics.CategoricalAccuracy()
                    ]) 

history = model.fit(x_train, Y_train, batch_size=100, epochs=20
                            ,validation_split=0.20
                            # ,validation_data=(x_test,Y_test)
                            ,validation_freq=1
                            ,callbacks=cp_callback
                            ,shuffle=True)
model.summary()