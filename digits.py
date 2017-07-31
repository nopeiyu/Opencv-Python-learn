#!/usr/bin/env python

'''
SVM and KNearest digit recognition.

Sample loads a dataset of handwritten digits from 'digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
   digits.py
'''

import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
from common import clock, mosaic
from numpy.linalg import norm


SZ = 20 # size of each digit is SZ x SZ，图片尺寸
CLASS_N = 10#分类数量，这里为0-9共10个数字
DIGITS_FN = 'img/digits.png'#图片地址，图片为50行，100列共5000个手写数字

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]#list(50,100)，列表存储的为array(20,20)
    cells = np.array(cells)#array(50L, 100L, 20L, 20L)
    if flatten:
        cells = cells.reshape(-1, sy, sx)#array(5000L, 20L, 20L)
    return cells

def load_digits(fn):
    print 'loading "%s" ...' % fn
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)#创建结果标签，array(5000L,)
    return digits, labels

#矫正倾斜的图像
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((10, 10), np.int32)#根据预测结果和正确结果画出matrix图
    for i, j in zip(labels, resp):
        confusion[i, int(j)] += 1
    print 'confusion matrix:'
    print confusion
    print

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0#错误预测的数字标为红色
        vis.append(img)
    return mosaic(25, vis)
def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

#计算图像的 HOG 描述符，
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)#笛卡尔坐标转极坐标
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]#tuple
        #对每一个小方块计算它们的朝向直方图（16 个 bin），使用梯度的大小做权重。
        #np.bincount(x,weights,minlength),这里用向量大小做权重，角度做统计，前面已经将ang转换成0-15共16个整数，array(4,16)每一个数字都转换成了64个数字表示的特征值
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        
        hist = np.hstack(hists)#array(1,64)
        
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

if __name__ == '__main__':
    print __doc__
    digits, labels = load_digits(DIGITS_FN)

    print 'preprocessing...'
    # shuffle digits
    rand = np.random.RandomState(321)#相当于随机数种子（seed）
    shuffle = rand.permutation(len(digits))#生成0-4999这5000个数字的随机序列
    digits, labels = digits[shuffle], labels[shuffle]

    digits2 = map(deskew, digits)
    samples = preprocess_hog(digits2)

    train_n = int(0.9*len(samples))#训练：测试 = 9：1
    cv2.imshow('test set', mosaic(25, digits[train_n:]))
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
    
    print 'training KNearest...'
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    cv2.imshow('KNearest test', vis)

    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    cv2.imshow('SVM test', vis)
    
    print 'saving SVM as "digits_svm.dat"...'
    model.save('digits_svm.dat')

    cv2.waitKey(0)