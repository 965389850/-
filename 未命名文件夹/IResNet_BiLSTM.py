from re import A
from keras import (
    Input,
    backend as K,
)
from keras.layers import (
    Dense, 
    concatenate,
    LSTM,
    Bidirectional,
    GlobalAveragePooling2D,
    Flatten,
    Activation, 
    BatchNormalization,
    Conv2D, 
    Dropout, 
    MaxPooling2D,
    concatenate,
    add,
    GRU,
    Multiply,
    Layer,
    Attention,
    Conv1D
)

from keras.models import Model
import tensorflow as tf
import model.IResNet as rs

def Stem(a):
    print('input',a.shape)
    a = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(a)
    # a = BatchNormalization()(a)
    a = MaxPooling2D(pool_size=(3, 3), strides=1,padding = 'same')(a)  # 池化层
    a = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(a)
    # a = BatchNormalization()(a)
    a = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(a)
    # a = BatchNormalization()(a)

    print('Stem output',a.shape)
    return a

def Inception_ResNet_A(inp):
    x = Activation('relu')(inp)
    init = x
    print('input',init.shape)
    
    b = Conv2D(filters=8, kernel_size=(1, 1), activation='relu', padding='same')(init)
    # b = BatchNormalization()(b)

    c = Conv2D(filters=8, kernel_size=(1, 1), activation='relu', padding='same')(init)
    # c = BatchNormalization()(c)
    c = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(c)
    # c = BatchNormalization()(c)

    d = Conv2D(filters=8, kernel_size=(1, 1), activation='relu', padding='same')(init)
    # d = BatchNormalization()(d)
    d = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(d)
    # d = BatchNormalization()(d)
    d = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(d)
    # d = BatchNormalization()(d)

    ir_merge = concatenate([b, c, d], axis=-1)
    e = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(ir_merge)
    out = add([init, e])
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)
    print('Inception_ResNet_A output',out.shape)

    return out

def Reduction_A(inp):
    init = inp
    print('input',init.shape)

    a = MaxPooling2D((3,3), strides=2,padding='valid')(init)
    print(a.shape)

    b = Conv2D(filters=8, kernel_size=(3,3), strides=2, activation='relu', padding='valid')(init)
    # b = BatchNormalization()(b)
    print(b.shape)

    c = Conv2D(filters=4, kernel_size=(1,1), activation='relu', padding='same')(init)
    # c = BatchNormalization()(c)
    c = Conv2D(filters=6, kernel_size=(3,3), activation='relu', padding='same')(c)
    # c = BatchNormalization()(c)
    c = Conv2D(filters=8, kernel_size=(3,3), strides=2,activation='relu', padding='valid')(c)
    # c = BatchNormalization()(c)
    print(c.shape)

    d = concatenate([a, b, c], axis=-1)
    # d = BatchNormalization()(d)
    # d = Activation('relu')(d)
    print('Reduction_A output',d.shape)
    return d

def Inception_ResNet_B(inp):
    init = inp
    print(init.shape)

    a = Conv2D(filters=8, kernel_size=(1,1), activation='relu', padding='same')(init)
    # a = BatchNormalization()(a)
    print('a.shape',a.shape)

    b = Conv2D(filters=8, kernel_size=(1,1), activation='relu', padding='same')(init)
    # b = BatchNormalization()(b)
    b = Conv2D(filters=8, kernel_size=(1,5), activation='relu', padding='same')(b)
    # b = BatchNormalization()(b)
    b = Conv2D(filters=8, kernel_size=(5,1), activation='relu', padding='same')(b)
    # b = BatchNormalization()(b)
    print('b.shape',b.shape)

    con = concatenate([a, b], axis=-1)
    print('c.shape',con.shape)
    c = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(con)
    # c = BatchNormalization()(c)
    
    out = add([init, c])
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)  
    print('Inception_ResNet_B output',out.shape)
    return out  

def Reduction_B(inp):
    init = inp
    print('input',init.shape)

    a = MaxPooling2D((3,3), strides=2, padding='valid')(init)
    print('a.shape',a.shape)

    b = Conv2D(filters=8, kernel_size=(1,1), activation='relu', padding='same')(init)
    # b = BatchNormalization()(b)
    b = Conv2D(filters=16, kernel_size=(3,3), strides=2, activation='relu', padding='valid')(b)
    # b = BatchNormalization()(b)
    print('b.shape',b.shape)

    c = Conv2D(filters=8, kernel_size=(1,1), activation='relu', padding='same')(init)
    # c = BatchNormalization()(c)
    c = Conv2D(filters=16, kernel_size=(3,3), strides=2, activation='relu', padding='valid')(c)
    # c = BatchNormalization()(c)
    print('c.shape',c.shape)

    d = Conv2D(filters=8, kernel_size=(1,1), activation='relu', padding='same')(init)
    # d = BatchNormalization()(d)
    d = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(d)
    # d = BatchNormalization()(d)
    d = Conv2D(filters=16, kernel_size=(3,3), strides=2, activation='relu', padding='valid')(d)
    # d = BatchNormalization()(d)
    print('d.shape',d.shape)

    e = concatenate([a, b, c, d], axis=-1)
    print('e.shape',e.shape)
    # e = BatchNormalization()(e)
    return e



def IResNet_BiLSTM(lstm1 = 128,lstm2 = 128, dense1 = 128,mod = 'softmax2'): #256,256,64
    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)
    cnninputs = tf.reshape(inputs,(-1,9,9,1))
    

    #resnet
    a = rs.Stem(cnninputs)
    print('stem')
    a = rs.Inception_ResNet_A(a)
    print('irA')
    a = rs.Reduction_A(a)
    print('rA')
    a = rs.Inception_ResNet_B(a)
    print('irB')
    a = rs.Reduction_B(a)
    print('rB')
    a = GlobalAveragePooling2D()(a)
    print(a.shape)


    outputs1 = a
    print('outputs1.shape',outputs1.shape)




    lstminputs = tf.reshape(inputs,(-1,1,81))

    # BiLSTM层
    b = lstminputs

    b = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(b)
    b = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(b)

    b = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(b)
    b = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh')(b)
    print('b.shape',b.shape)



    
    outputs2 = Flatten()(b)
    print('outputs2.shape',outputs2.shape)

    out = concatenate([outputs1, outputs2], axis=-1)
    print('out.shape',out.shape)


    out = Dense(dense1,activation='relu')(out)

    out = Dropout(0.5)(out)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(out)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(out)
    elif mod == 'softmax10':
        out = Dense(10,activation='softmax')(out)
    elif mod == 'softmax15':
        out = Dense(15,activation='softmax')(out)


    model = Model(inputs=[inputs], outputs=out)
    return (model)


def IResNet():
    inputs = Input(shape=(81))

    a = tf.reshape(inputs,(-1,9,9,1))
    print('stem')
    a = rs.Inception_ResNet_A(a)
    print('irA')
    a = rs.Reduction_A(a)
    print('rA')
    a = rs.Inception_ResNet_B(a)
    print('irB')
    a = rs.Reduction_B(a)
    print('rB')
    # a = Flatten()(a)
    # a = tf.reshape(a,(-1,1,4,80))
    a = GlobalAveragePooling2D()(a)
    a = Dropout(0.25)(a)
    # a = tf.keras.layers.Attention()([a,a])
    # a = Dense(128,activation='relu')(a)
    a = Dense(64,activation='relu')(a)
    a = Dropout(0.5)(a)
    a = Dense(2,activation='softmax')(a)

    outputs = a
    model = Model(inputs=[inputs], outputs=outputs)
    return model

def ATLG(i):
    x = 196
    inputs = Input(shape=(x))
    a = tf.reshape(inputs,(-1,1,x))
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(i, input_dim=x,recurrent_activation='sigmoid', activation = 'tanh'
    ,return_sequences=True
    )(a)
    # a = Bidirectional(LSTM(i, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True))(a)
    # a = Dropout(0.2)(a)
    # a = LSTM(i,input_dim=x,recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    # a = Bidirectional(LSTM(i, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True))(a)
    # a = Dropout(0.2)(a)

    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a) 
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)            
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)            
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(i, recurrent_activation='sigmoid',activation='tanh'
    # ,return_sequences=True
    )(a)
    # a = Dropout(0.5)(a)
    # print(a.shape)

    # a = tf.reshape(a,(-1,i,1))

    # a = tf.keras.layers.Attention()([a,a])
    # a = Dense(128,activation='relu')(a)
    # a = Attention()([a,a])
    # from attention import Attention
    # a = Attention(i)(a)
    # a = self_attention(i)(a)
    # a = Att(i,a,'attention')
    # a = Dropout(0.5)(a)
    # a = Dense(128,activation='relu')(inputs)
    # a = Dropout(0.5)(a)
    # a = Dense(128,activation='relu')(a)
    # a = Dropout(0.5)(a)
    # a = Dense(128,activation='relu')(a)
    # a = Dropout(0.5)(a)
    # a = Dense(128,activation='relu')(a)
    # a = Dropout(0.5)(a)
    # a = Flatten()(a)
    a = Dense(64,activation='relu')(a)
    a = Dropout(0.5)(a)
    a = Dense(10,activation='softmax')(a)

    outputs = a
    model = Model(inputs=[inputs], outputs=outputs)
    return model


def Att(att_dim,inputs,name):
    V = inputs
    QK = Dense(att_dim,use_bias = False)(inputs)
    QK = Activation("softmax",name=name)(QK)
    MV = Multiply()([V, QK])
    return(MV)


def mutresblock(a):
    b1 = Conv1D(filters=196, kernel_size=1, strides=1, padding='same')(a)
    b2 = Conv1D(filters=196, kernel_size=3, strides=1, padding='same')(a)
    b3 = Conv1D(filters=196, kernel_size=5, strides=1, padding='same')(a)
    b = add([b1,b2,b3])
    b = BatchNormalization()(b)
    b = Activation('relu')(b)
    b = LSTM(196,return_sequences=True)(b)
    b = add([b,a])
    b = BatchNormalization()(b)
    b = Activation('relu')(b)
    b = Dropout(0.5)(b)
    return b


def ABLIR(num=5,mod = 'softmax5'):

    K.clear_session()
    inputs = Input(shape=(196))  # 输入层
    print('inputs',inputs.shape)

    # a = tf.reshape(inputs,(-1,1,196))
    # a = tf.reshape(inputs,(-1,14,14,1))
    # i = 0
    # while i < num:
    #     a = mutresblock(a)
    #     i += 1
    a = Dense(256,activation='relu')(inputs)
    a = Dense(256,activation='relu')(a)
    a = Dense(256,activation='relu')(a)
    # a = LSTM(256,activation='tanh')(a)
    # a = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(a)
    # print(a.shape)
    # a = LSTM(256,activation='tanh')(a)
    # a = Flatten()(a)
    # a = Dense(64,activation='relu')(a)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(a)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(a)
    elif mod == 'softmax5':
        out = Dense(10,activation='softmax')(a)


    model = Model(inputs=[inputs], outputs=out)
    return (model)

def myLSTM(mod = 'softmax2'):

    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)

    a = tf.reshape(inputs,(-1,1,81))
    a = LSTM(256,activation='tanh')(a)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(a)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(a)
    elif mod == 'softmax5':
        out = Dense(10,activation='softmax')(a)

    model = Model(inputs=[inputs], outputs=out)
    return (model)

def CNN(mod = 'softmax2'):

    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)

    a = tf.reshape(inputs,(-1,9,9,1))
    a = Conv2D(256,kernel_size = (3,3),activation='relu',padding='same')(a)
    a = Flatten()(a)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(a)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(a)
    elif mod == 'softmax5':
        out = Dense(10,activation='softmax')(a)

    model = Model(inputs=[inputs], outputs=out)
    return (model)

def CNN_LSTM(mod = 'softmax2'):

    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)

    a = tf.reshape(inputs,(-1,9,9,1))
    a = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(a)
    a = LSTM(256,activation='tanh')(a)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(a)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(a)
    elif mod == 'softmax5':
        out = Dense(10,activation='softmax')(a)

    model = Model(inputs=[inputs], outputs=out)
    return (model)

def DNN(mod = 'softmax2'):

    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)

    a = Dense(256,activation='relu')(inputs)
    a = Dense(256,activation='relu')(a)
    a = Dense(64,activation='relu')(a)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(a)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(a)
    elif mod == 'softmax5':
        out = Dense(10,activation='softmax')(a)

    model = Model(inputs=[inputs], outputs=out)
    return (model)

def new_IRLG(lstm1 = 32,lstm2 = 256, dense1 = 64,mod = 'softmax2'):
    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)
    cnninputs = tf.reshape(inputs,(-1,9,9,1))
    

    #resnet
    a = rs.Stem(cnninputs)
    print('stem')
    a = rs.Inception_ResNet_A(a)
    print('irA')
    a = rs.Reduction_A(a)
    print('rA')
    a = rs.Inception_ResNet_B(a)
    print('irB')
    a = rs.Reduction_B(a)
    print('rB')
    a = GlobalAveragePooling2D()(a)
    print(a.shape)
    # a = Dropout(0.25)(a)


    a = tf.reshape(a,(-1,1,80))

    # BiLSTM层

    # b = tf.keras.layers.Attention()([b,b])
    # b = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(b)
    # b = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(b)
    # b = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(b)
    a = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    a = LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(a)
    # b = Bidirectional(LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True))(b)
    # b = LSTM(lstm1, activation='tanh', return_sequences=True)(b)
    # b = LSTM(lstm1, recurrent_activation='sigmoid', activation='tanh'
    # # ,return_sequences=True
    # )(b)
    # b = Bidirectional(LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True))(b)
    # b = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(b)
    # b = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(b)
    # b = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(b)
    a = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(a)
    a = GRU(lstm2, recurrent_activation='sigmoid',activation='tanh')(a)


    # b = Dense(lstm2, activation='relu')(b)

    # b = Attention()([b,b],use_causal_mask=False)
    # b = LSTM(lstm1, activation='tanh', return_sequences=True)(b)
    # b = LSTM(lstm1, activation='tanh', return_sequences=True)(b)
    # b = LSTM(lstm2, activation='tanh')(b)
    # b = Dropout(0.5)(b)
    # b = Dropout(0.2)(b)

    
    a = Flatten()(a)
    print('outputs2.shape',a.shape)

    out = Dense(dense1,activation='relu')(a)
    # out = Dense(dense1,activation='relu')(out)
    # out = Dense(dense1,activation='relu')(out)
    # out = Dense(dense1,activation='relu')(out)
    # out = Dense(dense1,activation='relu')(out)
    # out = Dropout(0.25)(out)
    # out = Dense(dense2,activation='relu')(out)
    out = Dropout(0.5)(out)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(out)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(out)
    elif mod == 'softmax10':
        out = Dense(10,activation='softmax')(out)
    elif mod == 'softmax15':
        out = Dense(15,activation='softmax')(out)


    model = Model(inputs=[inputs], outputs=out)
    return (model)

def test_IRLG(mod = 'softmax2'):
    K.clear_session()
    inputs = Input(shape=(81))  # 输入层
    print('inputs',inputs.shape)
    cnninputs = tf.reshape(inputs,(-1,9,9,1))
    

    #resnet
    a = rs.Stem(cnninputs)

    a = rs.Inception_ResNet_A(a)
    #(9,9,16)
    ra = a 

    a = rs.Reduction_A(a)
    #(4,4,32)
 
    a = rs.Inception_ResNet_B(a)
    #(4,4,32)
    rb = a

    a = rs.Reduction_B(a)
    #(4,4,32)

    a = GlobalAveragePooling2D()(a)
    #(80)
 
    # a = Dropout(0.25)(a)


    # a = tf.reshape(a,(-1,1,80))

    # BiLSTM层

    # a = Attention()([a,a],use_causal_mask=False)
    # b = Bidirectional(LSTM(lstm1, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True))(b)

    #model1
    ra = GlobalAveragePooling2D()(ra) #(16)
    ra1 = tf.reshape(ra,(-1,1,16))
    ra1 = LSTM(8, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(ra1)
    ra1 = GRU(16, recurrent_activation='sigmoid',activation='tanh')(ra1)
    ra1 = Flatten()(ra1)
    mod1out = add([ra,ra1]) #(32)
    # mod1out = Dense(16,activation = 'relu')(mod1out)

    # model2
    rb = GlobalAveragePooling2D()(rb) #(32)
    rb1 = tf.reshape(rb,(-1,1,32))
    rb1 = LSTM(16, recurrent_activation='sigmoid',activation = 'tanh', return_sequences=True)(rb1)
    rb1 = GRU(32, recurrent_activation='sigmoid',activation='tanh')(rb1)
    rb1 = Flatten()(rb1)
    mod2out = add([rb,rb1]) #(64)

    out = concatenate([a, mod1out, mod2out], axis=-1) #(96+80=176)
    # out = Dense(128,activation = 'relu')(out)
    # out = Dense(64,activation = 'relu')(out)



    print('outputs2.shape',a.shape)

    out = Dropout(0.5)(out)
    if mod == 'sigmoid':
        out = Dense(1,activation='sigmoid')(out)
    elif mod == 'softmax2':
        out = Dense(2,activation='softmax')(out)
    elif mod == 'softmax10':
        out = Dense(10,activation='softmax')(out)
    elif mod == 'softmax15':
        out = Dense(15,activation='softmax')(out)


    model = Model(inputs=[inputs], outputs=out)
    return (model)

