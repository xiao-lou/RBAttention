import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import TensorBoard
from loadClinicDBData import loadDataSet
from RBAttention import RBAttention


def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection/union


def saveModel(model):
    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass

    fp = open('models/modelP.json', 'w')
    fp.write(model_json)  # save model
    model.save_weights('models/modelW.h5')  # save model_weights


def evaluateModel(model, X_test, Y_test, batchSize):
    try:
        os.makedirs('results')
    except:
        pass


    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp, 0)

    for i in range(122):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1, 3, 2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0], Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1, 3, 3)
        plt.imshow(yp[i].reshape(yp[i].shape[0], yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection
        jacard = (np.sum(intersection) / np.sum(union))
        plt.suptitle('Jacard Index' + str(np.sum(intersection)) + '/' + str(np.sum(union)) + '=' + str(jacard))

        plt.savefig('results/' + str(i) + '.png', format='png')
        plt.close()

    jacard = 0
    dice = 0

    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()  # 预测值segmentation result
        y2 = Y_test[i].ravel()  # 真实值ground truth

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection) / np.sum(union))

        dice += (2. * np.sum(intersection)) / (np.sum(yp_2) + np.sum(y2))

    jacard /= len(Y_test)
    dice /= len(Y_test)
    print('Jacard Index : ' + str(jacard))
    print('Dice Coefficient : ' + str(dice))

    fp = open('models/log.txt', 'a')
    fp.write(str(jacard) + '\n')
    fp.close()

    fp = open('models/best.txt', 'r')
    best = fp.read()
    fp.close()

    if (jacard > float(best)):
        print('***********************************************')
        print('Jacard Index improved from ' + str(best) + ' to ' + str(jacard))
        print('***********************************************')
        fp = open('models/best.txt', 'w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)


def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):

#     model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=epochs, verbose=1,
#               callbacks=[TensorBoard(log_dir='./logs')])

    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch + 1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)
    
        evaluateModel(model, X_test, Y_test, batchSize)

    return model


if __name__ == '__main__':
    model = RBAttention(height=192, width=256, n_channels=3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
    saveModel(model)
    fp = open('models/log.txt', 'w')
    fp.close()
    fp = open('models/best.txt', 'w')
    fp.write('-1.0')
    fp.close()
    X_train, Y_train, X_test, Y_test = loadDataSet()
    trainStep(model, X_train, Y_train, X_test, Y_test, epochs=150, batchSize=4)
