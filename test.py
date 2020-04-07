import os
from model import SRCNN
from utils import load_test
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=None)
parser.add_argument('--label_size', type=int, default=None)
parser.add_argument('--c_dim', type=int, default=1)
parser.add_argument('--scale', type=int, default=3)

def main(args):
    srcnn = SRCNN(
        image_size=args.image_size,
        c_dim=args.c_dim,
        is_training=False)
    X_pre_test, X_test, Y_test, color = load_test(scale=args.scale)
    predicted_list = []
    for img in X_test:
        predicted = srcnn.process(img.reshape(1,img.shape[0],img.shape[1],1))
        predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],1))
    n_img = len(predicted_list)
    dirname = './result'
    for i in range(n_img):
        imgname = 'image{:02}'.format(i)
        # print(color[i].shape)
        # print(predicted_list[i].clip(min=0, max=255))
        # print(np.concatenate((predicted_list[i] / 255.0, color[i] / 255.0), axis=2))
        # cv2.imshow('result', X_test[i])
        # cv2.imshow('result', cv2.cvtColor(np.concatenate((Y_test[i], color[i]), axis=2), cv2.COLOR_YCrCb2BGR))
        # cv2.imshow('result', cv2.cvtColor(np.concatenate((predicted_list[i] / 255.0, color[i] / 255.0), axis=2).astype(np.float32), cv2.COLOR_YCrCb2BGR))
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(dirname,imgname+'_original.bmp'), X_pre_test[i])
        cv2.imwrite(os.path.join(dirname,imgname+'_input.bmp'), cv2.cvtColor(np.concatenate((X_test[i], color[i]), axis=2), cv2.COLOR_YCrCb2BGR))
        cv2.imwrite(os.path.join(dirname,imgname+'_answer.bmp'), cv2.cvtColor(np.concatenate((Y_test[i], color[i]), axis=2), cv2.COLOR_YCrCb2BGR))
        float_predicted = (predicted_list[i] / 255.0).clip(min=0., max=1.).astype(np.float64)
        normalized_predicted = np.expand_dims(cv2.normalize(float_predicted, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1), axis=2)
        print(normalized_predicted.shape)
        cv2.imwrite(os.path.join(dirname,imgname+'_predicted.bmp'), cv2.cvtColor(np.concatenate((normalized_predicted, color[i]), axis=2), cv2.COLOR_YCrCb2BGR))

if __name__ == '__main__':
    main(args=parser.parse_args())
