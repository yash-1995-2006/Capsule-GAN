import os, time, itertools, imageio
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
from utils import *
from CapsGenAV2 import capsgen
from sklearn.neighbors import NearestNeighbors
import pandas as pd

np.set_printoptions(threshold=np.inf)

# training parameters
batch_size = 64
lr = 0.005
train_epoch = 30

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        return capsgen(x)


def discriminator(input, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        print(x)
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)
        print(lrelu1)
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        print(lrelu2)
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        print(lrelu3)
        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(1, 1), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        flatconv4 = tf.reshape(lrelu4, [-1, 4 * 4 * 1024])
        print(lrelu4)

        Wh = tf.Variable(initial_value=tf.random_normal([4 * 4 * 1024, 1000]))
        Wo = tf.Variable(initial_value=tf.random_normal([1000, 2]))

        Bh = tf.Variable(initial_value=tf.random_normal([1000]))
        Bo = tf.Variable(initial_value=tf.random_normal([2]))

        hidden = tf.add(tf.matmul(flatconv4,Wh), Bh)
        out = tf.add(tf.matmul(hidden,Wo), Bo)
        softOut = tf.nn.softmax(out, axis=1)
        #print(o)
        return softOut, out



fixed_z_ = np.random.normal(0, 1, (64, 32))

def show_result(num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    ### Loading into pickle###
    path2 = path[:-4] + '.pickle'
    # f = open(path2, 'w+')
    test_images.dump(path2)
    # f.close()

    plt.ioff()
    size_figure_grid = 8
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(8, 8))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()

        ax[i, j].imshow(np.reshape(test_images[k], (32, 32)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()




def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()






# variables : input
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
z = tf.placeholder(tf.float32, shape=(None, 32))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z, W16, C16, B16, W8, C8, B8 = generator(z, isTrain)
flatG_z = tf.reshape(G_z, [batch_size,-1])
# networks : discriminator

D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# Wasserstein loss for each network
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]
print("Traning Vars: ", T_vars)


# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss)



# open session and initialize all variables
config = tf.ConfigProto()
sess = tf.InteractiveSession(config=config)
writer = tf.summary.FileWriter('./my_graph/mnist', sess.graph)
tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [32, 32]).eval()
train_set = (train_set - 0.5) / 0.5

test_set = tf.image.resize_images(mnist.test.images, [32, 32]).eval()
test_set = (test_set - 0.5) / 0.5

val_set = tf.image.resize_images(mnist.validation.images, [32, 32]).eval()
val_set = (val_set - 0.5) / 0.5




# results save folder
root = 'MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()


C1W = open("Capsule_Values/Capsule 1 W.txt","w+")
C1C = open("Capsule_Values/Capsule 1 C.txt","w+")
C1B = open("Capsule_Values/Capsule 1 B.txt","w+")

C2W = open("Capsule_Values/Capsule 2 W.txt","w+")
C2C = open("Capsule_Values/Capsule 2 C.txt","w+")
C2B = open("Capsule_Values/Capsule 2 B.txt","w+")

X, Y = train_set, mnist.train.labels
X, Y = np.vstack((X, test_set)), np.vstack((Y,mnist.test.labels))
X, Y = np.vstack((X, val_set)), np.vstack((Y,mnist.validation.labels))
X, Y = X.reshape((70000, -1)), Y.reshape((70000,-1))
nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
nn.fit(X)


loc = 0
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(mnist.train.num_examples // batch_size):

        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 32))
        loss_d_ = 0
        for v in range(5):
            loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        #z_ = np.random.normal(0, 1, (batch_size, 32))
        loss_g_, _,l1W, l1C, l1B, l2W, l2C, l2B, generatedImages = sess.run([G_loss, G_optim, W16, C16, B16, W8, C8, B8, flatG_z], {z: z_, x: x_, isTrain: True})
        G_losses.append(loss_g_)

        print('Batch Done: ', iter + 1)
        if iter%100 == 0:
            distance, index = nn.kneighbors(generatedImages)
            labels = Y[index]
            for l in range(batch_size):
                print('Closest label: ', labels[l], ' distance: ', distance[l])


        '''
        if iter%50 == 0:
                
            str1 = "Iter: " + str(iter + 1) + "\n" + np.array2string(np.ravel(l1W)).replace('\n', '') + "\n"
            str2 = "Iter: " + str(iter + 1) + "\n" + np.array2string(np.ravel(l1C)).replace('\n', '') + "\n"
            str3 = "Iter: " + str(iter + 1) + "\n" + np.array2string(np.ravel(l2W)).replace('\n', '') + "\n"
            str4 = "Iter: " + str(iter + 1) + "\n" + np.array2string(np.ravel(l2C)).replace('\n', '') + "\n"
            str5 = "Iter: " + str(iter + 1) + "\n" + np.array2string(np.ravel(l1B)).replace('\n', '') + "\n"
            str6 = "Iter: " + str(iter + 1) + "\n" + np.array2string(np.ravel(l2B)).replace('\n', '') + "\n"

            C1W.write(str1)
            C1C.write(str2)
            C1B.write(str5)
            C2W.write(str3)
            C2C.write(str4)
            C2B.write(str6)
            
            fl1W, fl1C, fl1B, fl2W, fl2C, fl2B = np.ravel(l1W), np.ravel(l1C), np.ravel(l1B), np.ravel(l2W), np.ravel(l2C), np.ravel(l2B)
            try:
                loc += 1
                dfl1W.loc[loc] = np.expand_dims(fl1W, axis=0)
                dfl1C.loc[loc] = np.expand_dims(fl1C, axis=0)
                dfl1B.loc[loc] = np.expand_dims(fl1B, axis=0)
                dfl2W.loc[loc] = np.expand_dims(fl2W, axis=0)
                dfl2C.loc[loc] = np.expand_dims(fl2C, axis=0)
                dfl2B.loc[loc] = np.expand_dims(fl2B, axis=0)
                dfl1C.loc[loc] = fl1C
                dfl1B.loc[loc] = fl1B
                dfl2C.loc[loc] = fl2C
                dfl2B.loc[loc] = fl2B
            except:
                dfl1W = pd.DataFrame(np.expand_dims(fl1W, axis=0))
                dfl1C = pd.DataFrame(np.expand_dims(fl1C, axis=0))
                dfl1B = pd.DataFrame(np.expand_dims(fl1B, axis=0))
                dfl2W = pd.DataFrame(np.expand_dims(fl2W, axis=0))
                dfl2C = pd.DataFrame(np.expand_dims(fl2C, axis=0))
                dfl2B = pd.DataFrame(np.expand_dims(fl2B, axis=0))

            dfl1W.to_csv("l1W.csv")
            dfl1C.to_csv("l1C.csv")
            dfl1B.to_csv("l1B.csv")
            dfl2W.to_csv("l2W.csv")
            dfl2C.to_csv("l2C.csv")
            dfl2B.to_csv("l2B.csv")
        '''


    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result(epoch + 1, save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

'''
C1W.close()
C2W.close()
C1C.close()
C2C.close()
'''

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
writer.close()
sess.close()
