from nets import resnet_v2
import tensorflow as tf

from tensorflow.contrib import slim
import utils



def inception(cv,out_put,name,stride):
    with tf.variable_scope(name):
        cv8 = slim.conv2d(cv, 256, kernel_size=1, stride=1)
        c1 = slim.conv2d(cv8,out_put/4,kernel_size=1,stride=stride)
        c2 = slim.conv2d(cv8, out_put/4, kernel_size=3, stride=stride)
        c3 = slim.conv2d(cv8, out_put/4, kernel_size=5, stride=stride)
        c = tf.concat([c1,c2,c3],axis=3)
        c = slim.conv2d(c,out_put/2,1,1)

    return c

def get_mask_fp(Mixed_3c ,Mixed_4e,Mixed_5c):
    x = slim.conv2d_transpose(Mixed_5c,256,kernel_size=2,stride=2)
    x = tf.concat([x,Mixed_4e],axis=3)
    x = slim.conv2d(x,512,1,1)
    x = slim.conv2d_transpose(x,256,kernel_size=2,stride=2)
    x = tf.concat([x, Mixed_3c], axis=3)
    x = slim.conv2d(x, 256, 1, 1)
    return x
def model(img,cfg):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_50(img,  is_training=True)

    base_64 = end_points['resnet_v2_50/block1']
    base_32 = end_points['resnet_v2_50/block2']
    base_16_0 = end_points['resnet_v2_50/block3']
    base_16_1 = end_points['resnet_v2_50/block4']
    vbs = slim.get_trainable_variables()
    # vbs = None
    base_16_0 = slim.conv2d(base_16_0, 512, 1)
    base_16_1 = slim.conv2d(base_16_1, 512, 1)
    base_16 = tf.concat([base_16_0,base_16_1],axis=3)

    mask_fp = get_mask_fp(base_64, base_32, base_16)

    cell_11 = tf.image.resize_bilinear(base_16,
                                       size=[int(32 * (cfg.image_size / 512)), int(32 * (cfg.image_size / 512))])
    cell_11 = tf.concat([cell_11, base_32], axis=3)

    cell_7 = tf.image.resize_bilinear(base_32,
                                      size=[int(64 * (cfg.image_size / 512)), int(64 * (cfg.image_size / 512))])
    cell_7 = tf.concat([cell_7, base_64], axis=3)


    cell_11 = slim.conv2d(cell_11, 1024, kernel_size=1, activation_fn=slim.nn.relu)

    cell_7 = slim.conv2d(cell_7, 512, kernel_size=3, activation_fn=slim.nn.relu)
    cell_7 = slim.conv2d(cell_7, 256, kernel_size=1, activation_fn=slim.nn.relu)

    cv6 = slim.conv2d(cell_11, 1024, kernel_size=3, rate=6, activation_fn=slim.nn.relu, scope='conv6')
    cv7 = slim.conv2d(cv6, 1024, kernel_size=1, activation_fn=slim.nn.relu, scope='conv7')

    s = utils.normalize_to_target(cell_7, target_norm_value=cfg.norm_value, dim=1)

    #cv8 = inception(cv7, out_put=512, name='cv8', stride=2)
    #cv9 = inception(cv8, out_put=256, name='cv9', stride=2)
    #cv10 = inception(cv9, out_put=256, name='cv10', stride=2)
    #cv11 = inception(cv10, out_put=256, name='cv11', stride=2)

    cv8 = slim.conv2d(cv7, 512, kernel_size=3, stride=1,rate=4, scope='conv8_2')
    cv8 = slim.conv2d(cv8, 256, kernel_size=1, stride=1, scope='conv8_0')
    cv8 = slim.conv2d(cv8, 512, kernel_size=3, stride=2, scope='conv8_1')

    cv9 = slim.conv2d(cv8, 256, kernel_size=3, stride=1, rate=2,scope='conv9_2')
    cv9 = slim.conv2d(cv9, 128, kernel_size=1, stride=1, scope='conv9_0')
    cv9 = slim.conv2d(cv9, 256, kernel_size=3, stride=2, scope='conv9_1')

    cv10 = slim.conv2d(cv9, 128, kernel_size=1, stride=1, scope='conv10_0')
    cv10 = slim.conv2d(cv10, 256, kernel_size=3, stride=2, scope='conv10_1')

    cv11 = slim.conv2d(cv10, 128, kernel_size=1, stride=1, scope='conv11_0')
    cv11 = slim.conv2d(cv11, 256, kernel_size=3, stride=2, scope='conv11_1')


    #64,32,16,8,4,2
    source = [s, cv7, cv8, cv9, cv10, cv11]
    conf = []
    loc = []
    for cv, num in zip(source, cfg.Config['aspect_num']):
        print(cv)
        loc.append(slim.conv2d(cv, num * 4, kernel_size=3, stride=1, activation_fn=None))

        conf.append(
            slim.conv2d(cv, num * cfg.Config['num_classes'], kernel_size=3, stride=1, activation_fn=None))

    loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
    conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf],
                     axis=1)
    print(loc)
    return loc, conf, mask_fp, vbs


