# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
import json
import pickle
import zipfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow
import jieba
import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import (load_img, img_to_array)
from tqdm import tqdm

from config import img_rows, img_cols
from config import start_word, stop_word, unknown_word
from config import train_annotations_filename
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_image_folder, valid_image_folder, test_a_image_folder, test_b_image_folder
from config import valid_annotations_filename

#调用Keras中的ResNet50模型，加载在ImageNet ILSVRC比赛中已经训练好的权重
#include_top表示是否包含模型顶部的全连接层，如果不包含，则可以利用这些参数来做一些定制的事情
image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

#确定是否存在文件夹
def ensure_folder(folder):
    #如果不存在文件夹，创建文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)

#解压文件
def extract(folder):
    #folder.zip
    filename = '{}.zip'.format(folder)
    #输出解压名称并执行解压操作
    #print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')

#将图像文件编码
def encode_images(usage,image_name):
    encoding = {}
    #编码训练集
    if usage == 'train':
        image_folder = train_image_folder
    #编码验证集
    elif usage == 'valid':
        image_folder = valid_image_folder
    #编码测试集a
    elif usage == 'test_a':
        image_folder = test_a_image_folder
    #编码测试集b
    else:  # usage == 'test_b':
        image_folder = test_b_image_folder
    #batch_size为256
    batch_size = 256
    #names储存文件夹中所有的jpg文件名称
    names = [image_name]
    print(names)
    #计算一共多少批次，ceil为向上取整
    num_batches = int(np.ceil(len(names) / float(batch_size)))

    #输出编码过程
    #print('ResNet50提取特征中...')
    #对每个batche进行处理，使用tqdm库显示处理进度
    for idx in range(num_batches):
      #该批次开始的位置
        i = idx * batch_size
            #该批次的长度，会出现最后一个批次不够batchsize的情况
        length = min(batch_size, (len(names) - i))
            #使用empty创建一个多维数组
        image_input = np.empty((length, img_rows, img_cols, 3))
            #对于每一张图片
        for i_batch in range(length):
                #提取图片名称
                #提取路径名称
            filename = names[0]
                #keras读取图片，并且将图片调整为224*224
            print(filename)
            img = load_img(filename, target_size=(img_rows, img_cols))
        
            #将图片转为矩阵
        img_array = img_to_array(img)
            #使用keras内置的preprocess_input进行图片预处理，默认使用caffe模式去均值中心化
        img_array = tensorflow.keras.applications.resnet50.preprocess_input(img_array)
            #将处理后的图片保存到image_input中
        image_input[i_batch] = img_array

        #使用ResNet50网络进行预测，预测结果保存到preds中
        preds = image_model.predict(image_input)

        #对于每一张图片
        for i_batch in range(length):
            #提取图片名称
            image_name = names[i + i_batch]
            #把预测结果保存到encoding中
            encoding[image_name] = preds[i_batch]

    #用相应的类别命名
    filename = 'data/encoded_{}_images.p'.format(usage)
    #使用python的pickle模块把数据进行序列化，把encoing保存到filename中
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)
    #print('ResNet50提取特征完毕...')

#处理数据集的标注部分，生成训练集的词库
def build_train_vocab():
    #提取训练集标注文件的路径
    #data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    #读取json格式的标注文件
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    #输出处理进度
    #print('building {} train vocab')
    #创建一个无序不重复元素集
    vocab = set()
    #使用tqdm输出进度
    for a in tqdm(annotations):
        #提取annotations每一行的caption注释
        caption = a['caption']
        #对于每一个caption
        for c in caption:
            #使用jieba进行分词
            seg_list = jieba.cut(c)
            #把每个词加入到vocab中
            for word in seg_list:
                vocab.add(word)
    #在vocab中加入<start><stop><UNK>
    vocab.add(start_word)
    vocab.add(stop_word)
    vocab.add(unknown_word)

    #将vocab写入vocab_train.p
    filename = 'data/vocab_train.p'
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(vocab, encoded_pickle)

#创建samples
def build_samples(usage):
    #如果进行训练
    if usage == 'train':
        #路径为train_folder
        annotations_path = os.path.join(train_folder, train_annotations_filename)
    else:
        #否则路径为valid_folder
        annotations_path = os.path.join(valid_folder, valid_annotations_filename)
    with open(annotations_path, 'r') as f:
        #同时加载json文件
        annotations = json.load(f)

    #将vocab文件反序列化提取词汇
    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    #index to word 对vocab进行排序
    idx2word = sorted(vocab)
    #word to index zip函数将idx2word与序号索引打包为元祖，用dict函数将映射关系构造为字典，词：索引
    word2idx = dict(zip(idx2word, range(len(vocab))))

    #输出进度信息
    #print('building {} samples'.format(usage))
    #列表samples
    samples = []
    #对于每一项annotation
    for a in tqdm(annotations):
        #提取image_id
        image_id = a['image_id']
        #提取caption
        caption = a['caption']
        #对于每一项caption
        for c in caption:
            #使用jieba进行分词
            seg_list = jieba.cut(c)
            #列表inpit
            input = []
            #last_word标签设置为start
            last_word = start_word
            #使用enumerate函数列出下标和数据
            for j, word in enumerate(seg_list):
                #如果词库中没有word
                if word not in vocab:
                    #word修改为UNK
                    word = unknown_word
                #input添加序号
                input.append(word2idx[last_word])
                #samples添加id，input，output
                samples.append({'image_id': image_id, 'input': list(input), 'output': word2idx[word]})
                #last_word设置成word
                last_word = word
            #input添加last_word
            input.append(word2idx[last_word])
            #samples添加id，input，stop_word
            samples.append({'image_id': image_id, 'input': list(input), 'output': word2idx[stop_word]})

    #打包samples信息
    filename = 'data/samples_{}.p'.format(usage)
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)

#!/usr/bin/env python
#coding:utf-8
"""a demo of matplotlib"""
import matplotlib as  mpl
from matplotlib.font_manager import FontProperties
myfont =  FontProperties(fname="./font/simhei.ttf",size=12)
# 为了服务器上的图像保存
mpl.use('Agg')
from matplotlib  import pyplot as plt
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False #解决负号‘-‘显示为方块的问题

import argparse
import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']  # for Windows

# plt.rcParams['font.sans-serif'] = ['simhei']

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
# from scipy.misc import imresize
from imageio import imread

import sys
sys.path.append('./')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # 已经被遗弃，更改为下面的写法
    # img = imresize(img, (256, 256))
    # img = skimage.transform.resize(img,(256,256))
    # BUG 很有可能导致了预测结果不同~ 以确定，就是这个导致的结果不同！！ 暂时带上后面的
    img = np.array(Image.fromarray(img).resize(size=(256,256),resample= Image.BILINEAR))

    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s) #changed / to // by jason
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, i, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    print(words)

    for t in range(len(words)):
        if t > 50:
            break
        # 向上取整,每行5个词，共 int(np.ceil(len(words) / 5.)) 行
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12, font = myfont)
        # 画图但是并不显示
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig('images/out_{}.jpg'.format(i))
    plt.close()

#================================循环查找路径=================
import time


def get_image_path(new_path):
    global predicted_path
    print("wait a new path...")
    # =========== 2.0 版本==============
    while(True):

        time.sleep(5)
        paths = {}
        with open(new_path,'r') as f:

            try:
                paths = json.load(f)
            except json.decoder.JSONDecodeError:
                print("no new path...")
                continue

            paths_list = list(paths.keys())
            # print(paths_list)

            for _path in paths_list:
                if paths[_path] is False and (_path not in predicted_path):
                    insert_path(_path)
                    return _path

# 创建固定长度的已预测缓存
predicted_path = []

def insert_path(predict_path):
    global predicted_path
    # 缓存长度200
    if len(predicted_path) == 200:
        predicted_path.pop(0)
        predicted_path.append(predict_path)
    else:
        predicted_path.append(predict_path)


# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence
#从config文件中引入一些参数 包括token最大长度 测试文件夹长度 最优的模型参数
from config import max_token_length, test_a_image_folder, best_model
from forward import build_model
from generated import test_gen

#使用训练好的模型对图片进行测试
def beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=3):
    start = [word2idx["<start>"]]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_token_length:
        temp = []
        for s in start_word:
		    #对序列进行填充的预处理，在其后添0，使其序列统一大小为max_token_length
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_token_length, padding='post')
			#每次取一个图片进行测试
            e = encoding_test[image_name]
			#使用模型对该图片进行测试
            preds = model.predict([np.array([e]), np.array(par_caps)])
            #从预测的结果中取前beam_index个
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
			# 创建一个新的list结构 将预测出的词和词的概率以组对的形式存储
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
        #将处理好的预测值赋值回start word
        start_word = temp
        # Sorting according to the probabilities
		# 根据概率排序
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
		#获得最有可能正确的词
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
	#根据id取出单词
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    #组合成句
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ''.join(final_caption[1:])
    return final_caption


import time


def get_image_path(new_path):
    global predicted_path
    print("wait a new path...")
    # =========== 2.0 版本==============
    while(True):

        time.sleep(5)
        paths = {}
        with open(new_path,'r') as f:

            try:
                paths = json.load(f)
            except json.decoder.JSONDecodeError:
                print("no new path...")
                continue

            paths_list = list(paths.keys())
            # print(paths_list)

            for _path in paths_list:
                if paths[_path] is False and (_path not in predicted_path):
                    insert_path(_path)
                    return _path





if __name__ == '__main__':
    #图片的channel为3
    channel = 3
    #设置模型权重的地址
    model_weights_path = os.path.join('models', best_model)
    print('Loading models')
	#创建模型
    model = build_model()
	#加载模型权重
    model.load_weights(model_weights_path)
    print('Models found')

    #print(model.summary())  
    #加载语料库
    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
	#将word转化为数字  方便输入网络 进行预测
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))
    #print('语料库加载完毕...')


    while(True):
        print("Waiting for a new image path")
        img_path = get_image_path("/var/www/html/share_path.json")
        print("image_path",img_path)

        encode_images('test_a',img_path)

        encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))
   
        names = [f for f in encoded_test_a.keys()]
     
        samples = names

        sentences = []
        for i in range(len(samples)):
	    #依次取图片
            image_name = samples[i]
            filename = os.path.join(test_a_image_folder, image_name)
        # # print('Start processing image: {}'.format(filename))
        # image_input = np.zeros((1, 2048))
        # image_input[0] = encoded_test_a[image_name]
        #
        # start_words = [start_word]
        # while True:
        #     text_input = [word2idx[i] for i in start_words]
        #     text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
        #     preds = model.predict([image_input, text_input])
        #     # print('output.shape: ' + str(output.shape))
        #     word_pred = idx2word[np.argmax(preds[0])]
        #     start_words.append(word_pred)
        #     if word_pred == stop_word or len(start_word) > max_token_length:
        #         break
        #使用beam_search机制进行预测

            candidate = beam_search_predictions(model, image_name, word2idx, idx2word, encoded_test_a,
                                            beam_index=1)
        #打印结果
            #print(candidate)
            sentences.append(candidate)
            print('Loading pictures')
        #读取图片 并调整其大小
            img = cv.imread(filename)   
            print('Loading complete')
        #img = cv.resize(img, (img_rows, img_cols), cv.INTER_CUBIC)
            if not os.path.exists('images'):
                os.makedirs('images')
            cv.imwrite('images/{}_image.jpg'.format(i), img)
    #将预测产生的描述信息输出到json文件中
        sentence = ''.join(sentences) #这里只可以对一个图像进行展示否则是两个组合的句子 因为考虑到getCaption的load
        content = {}
        with open("predict_result/predict.json", 'r+') as file:
            if bool(file.readline()):
                file.seek(0)
                content = json.load(file)
            file.seek(0)
            if not img_path in content:
                content[img_path] = {}
            content[img_path]['CN_SoftAtt'] = sentence
            json.dump(content,file)
            #print("CN_SoftAtt存储结果文件结束")
        K.clear_session()
