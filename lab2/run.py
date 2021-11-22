# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    x = import_module('models.' + model_name)
    #batch = [8, 16, 32, 64]
    drop_out=[0.1, 0.2, 0.3, 0.4, 0.5]
    #learning_rate=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    #hidden_size=[64, 128, 256]
    #embedding_size=[64, 128, 256]
    train_loss = np.zeros((5, 20), dtype=np.float32)
    test_loss = np.zeros((5, 20), dtype=np.float32)
    m = 0
    for i in drop_out:
        config = x.Config(dataset, embedding, i)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        start_time = time.time()
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        if model_name != 'Transformer':
            init_network(model)
        print(model.parameters)
        train_loss[m], test_loss[m] = train(config, model, train_iter, dev_iter, test_iter)
        m = m + 1
    f = range(1, 21)
    for n in range(0, 5):
        for m in range(0, 20):
            if train_loss[n][m]  == 0 :
                if m != 0 and m != 19:
                    train_loss[n][m] = (train_loss[n][m+1] + train_loss[n][m-1])/2
                elif m == 0 :
                    train_loss[n][m] = train_loss[n][m+1] + 0.1
                else:
                    train_loss[n][m] = train_loss[n][m-1]
    for n in range(3):
        for m in range(20):
            if test_loss[n][m]  == 0  :
                if m != 0 and m != 19:
                    test_loss[n][m] = (test_loss[n][m+1] + test_loss[n][m-1])/2
                elif m == 0 :
                    test_loss[n][m]  = test_loss[n][m+1]
                else:
                    test_loss[n][m] = test_loss[n][m-1]
    plt.subplot(121)
    loss1 = plt.plot(f, train_loss[0], color='#000249', linewidth=1.3, linestyle='-', label='batch_size = 8')
    loss2 = plt.plot(f, train_loss[1], color='#0F4392', linewidth=1.3, linestyle='-', label='batch_size = 16')
    loss3 = plt.plot(f, train_loss[2], color='#FF5151', linewidth=1.3, linestyle='-', label='batch_size = 32')
    loss4 = plt.plot(f, train_loss[3], color='#FF8B8B', linewidth=1.3, linestyle='-', label='batch_size = 64')
    loss5 = plt.plot(f, train_loss[4], color='#b30047', linewidth=1.3, linestyle='-', label='batch_size = 64')
    plt.title("Drop out(Train Set)")  # 设置标题及字体
    plt.xlabel(u"epochs")
    plt.ylabel(u"loss")
    plt.subplot(122)
    loss6 = plt.plot(f, test_loss[0], color='#000249', linewidth=1.3, linestyle='-', label='batch_size = 8')
    loss7 = plt.plot(f, test_loss[1], color='#0F4392', linewidth=1.3, linestyle='-', label='batch_size = 16')
    loss8 = plt.plot(f, test_loss[2], color='#FF5151', linewidth=1.3, linestyle='-', label='batch_size = 32')
    loss9 = plt.plot(f, test_loss[3], color='#FF8B8B', linewidth=1.3, linestyle='-', label='batch_size = 16')
    loss10 = plt.plot(f, test_loss[4], color='#b30047', linewidth=1.3, linestyle='-', label='batch_size = 32')
    plt.title("Drop out(Test Set)")  # 设置标题及字体
    plt.xlabel(u"epochs")
    plt.ylabel(u"loss")
    plt.show()


