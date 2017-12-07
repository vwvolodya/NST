import argparse
import os
import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from tqdm import tqdm as progressbar


class NST:
    def __init__(self, model_path, style_layers, target_layer, alpha=10, beta=40):
        self.model_path = model_path
        self.alpha = alpha
        self.beta = beta
        self.means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        self.target_layer = target_layer
        self.style_layers = style_layers
        self.output_dir = "output"
        self.initial_noise_ratio = 0.6
        self._model = None
        self.sess = None
        tf.reset_default_graph()

    def _load_vgg_model(self, h, w, c):
        """
        Returns a model for the purpose of 'painting' the picture.
        Takes only the convolution layer weights and wrap using the TensorFlow
        Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
        the paper indicates that using AveragePooling yields better results.
        The last few fully connected layers are not used.
        Here is the detailed configuration of the VGG model:
            0 is conv1_1 (3, 3, 3, 64)
            1 is relu
            2 is conv1_2 (3, 3, 64, 64)
            3 is relu
            4 is maxpool
            5 is conv2_1 (3, 3, 64, 128)
            6 is relu
            7 is conv2_2 (3, 3, 128, 128)
            8 is relu
            9 is maxpool
            10 is conv3_1 (3, 3, 128, 256)
            11 is relu
            12 is conv3_2 (3, 3, 256, 256)
            13 is relu
            14 is conv3_3 (3, 3, 256, 256)
            15 is relu
            16 is conv3_4 (3, 3, 256, 256)
            17 is relu
            18 is maxpool
            19 is conv4_1 (3, 3, 256, 512)
            20 is relu
            21 is conv4_2 (3, 3, 512, 512)
            22 is relu
            23 is conv4_3 (3, 3, 512, 512)
            24 is relu
            25 is conv4_4 (3, 3, 512, 512)
            26 is relu
            27 is maxpool
            28 is conv5_1 (3, 3, 512, 512)
            29 is relu
            30 is conv5_2 (3, 3, 512, 512)
            31 is relu
            32 is conv5_3 (3, 3, 512, 512)
            33 is relu
            34 is conv5_4 (3, 3, 512, 512)
            35 is relu
            36 is maxpool
            37 is fullyconnected (7, 7, 512, 4096)
            38 is relu
            39 is fullyconnected (1, 1, 4096, 4096)
            40 is relu
            41 is fullyconnected (1, 1, 4096, 1000)
            42 is softmax
        """

        vgg = scipy.io.loadmat(self.model_path)

        vgg_layers = vgg['layers']

        def _weights(layer, expected_layer_name):
            """
            Return the weights and bias from the VGG model for a given layer.
            """
            wb = vgg_layers[0][layer][0][0][2]
            W = wb[0][0]
            b = wb[0][1]
            layer_name = vgg_layers[0][layer][0][0][0][0]
            assert layer_name == expected_layer_name
            return W, b

        def _relu(conv2d_layer):
            """
            Return the RELU function wrapped over a TensorFlow layer. Expects a
            Conv2d layer input.
            """
            return tf.nn.relu(conv2d_layer)

        def _conv2d(prev_layer, layer, layer_name):
            """
            Return the Conv2D layer using the weights, biases from the VGG
            model at 'layer'.
            """
            W, b = _weights(layer, layer_name)
            W = tf.constant(W)
            b = tf.constant(np.reshape(b, b.size))
            return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

        def _conv2d_relu(prev_layer, layer, layer_name):
            """
            Return the Conv2D + RELU layer using the weights, biases from the VGG
            model at 'layer'.
            """
            return _relu(_conv2d(prev_layer, layer, layer_name))

        def _avgpool(prev_layer):
            """
            Return the AveragePooling layer.
            """
            return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Constructs the graph model.
        graph = {}
        graph['input'] = tf.Variable(np.zeros((1, h, w, c)), dtype='float32')
        graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = _avgpool(graph['conv1_2'])
        graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = _avgpool(graph['conv2_2'])
        graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = _avgpool(graph['conv3_4'])
        graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = _avgpool(graph['conv4_4'])
        graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = _avgpool(graph['conv5_4'])

        self._model = graph
        return graph

    @classmethod
    def _gram_matrix(cls, a):
        matrix = tf.matmul(a, tf.transpose(a))
        return matrix

    @classmethod
    def _layer_style_cost(cls, style, generated):
        m, h, w, c = generated.get_shape().as_list()
        style = tf.transpose(tf.reshape(style, [h * w, c]))
        generated = tf.transpose(tf.reshape(generated, [h * w, c]))
        gram_style = cls._gram_matrix(style)
        gram_generated = cls._gram_matrix(generated)
        layer_cost = tf.reduce_sum(tf.square(tf.subtract(gram_generated, gram_style))) / (4 * c ** 2 * (h * w) ** 2)
        return layer_cost

    @classmethod
    def _content_cost(cls, content, generated):
        m, h, w, c = generated.get_shape().as_list()
        content = tf.transpose(tf.reshape(content, [h * w, c]))
        generated = tf.transpose(tf.reshape(generated, [h * w, c]))
        cost = 1 / (4 * h * w * c) * tf.reduce_sum(tf.square(tf.subtract(content, generated)))
        return cost

    def _style_cost(self):
        style_cost = 0
        for name, weight in self.style_layers:
            output = self._model[name]
            style = self.sess.run(output)
            generated = output
            layer_cost = self._layer_style_cost(style, generated)
            style_cost += weight * layer_cost
        return style_cost

    def _weighted_cost(self, content_cost, style_cost):
        cost = self.alpha * content_cost + self.beta * style_cost
        return cost

    def _get_total_cost(self, content_image, style_image):
        self.sess.run(self._model['input'].assign(content_image))
        out = self._model[self.target_layer]
        content = self.sess.run(out)
        generated = out
        content_cost = self._content_cost(content, generated)

        self.sess.run(self._model['input'].assign(style_image))
        style_cost = self._style_cost()

        total_cost = self._weighted_cost(content_cost, style_cost)
        return total_cost

    def _generate_noise_image(self, content_image):
        _, h, w, c = content_image.shape
        noise_image = np.random.uniform(-20, 20, (1, h, w, c)).astype('float32')
        input_image = noise_image * self.initial_noise_ratio + content_image * (1 - self.initial_noise_ratio)
        return input_image

    @classmethod
    def _load(cls, content_image_path, style_image_path):
        content_image = scipy.misc.imread(content_image_path)
        style_image = scipy.misc.imread(style_image_path)
        c_h, c_w, _ = content_image.shape
        s_h, s_w, _ = style_image.shape
        new_shape = (min(c_h, s_h), min(c_w, s_w))
        style_image = scipy.misc.imresize(style_image, new_shape)
        content_image = scipy.misc.imresize(content_image, new_shape)
        return content_image, style_image

    def _prepare_image(self, image):
        image = np.reshape(image, ((1,) + image.shape))
        image = image - self.means
        return image

    def _save_image(self, path, image):
        full_path = os.path.join(self.output_dir, path)
        image = image + self.means
        image = np.clip(image[0], 0, 255).astype('uint8')
        scipy.misc.imsave(full_path, image)

    def transfer(self, optim, content_image_path, style_image_path, result_path=None, num_epochs=200):

        c_image, s_image = self._load(content_image_path, style_image_path)
        c_image = self._prepare_image(c_image)
        s_image = self._prepare_image(s_image)

        input_image = self._generate_noise_image(c_image)

        _, height, width, depth = c_image.shape
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self._load_vgg_model(height, width, depth)

        objective = self._get_total_cost(c_image, s_image)
        train_step = optim.minimize(objective)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self._model['input'].assign(input_image))

        generated_image = None
        for i in progressbar(range(num_epochs)):
            self.sess.run(train_step)
            generated_image = self.sess.run(self._model['input'])
            if i % 20 == 0:
                cost = self.sess.run([objective])
                print("\nIteration " + str(i) + " : total cost = " + str(cost) + "\n")
                self._save_image(str(i) + ".png", generated_image)

        if result_path is not None:
            self._save_image(result_path, generated_image)
        return generated_image


def main():
    parser = argparse.ArgumentParser(description='Prisma-like app to apply style for images.')
    parser.add_argument('-lr', action="store", dest="lr", type=float,
                        default=2.0, help="Learning rate for Adam optimizer")
    parser.add_argument('-t', action="store", dest="t",
                        default="conv4_2", help="Target layer used for comparison")
    parser.add_argument('-a', action="store", dest="a", type=int,
                        default=10, help="Blending parameter alpha")
    parser.add_argument('-b', action="store", dest="b", type=int,
                        default=40, help="Blending parameter beta")
    parser.add_argument('-e', action="store", dest="e", type=int,
                        default=400, help="Number of epochs to train")
    parser.add_argument('-m', action="store", dest="m",
                        default="pretrained/imagenet-vgg-verydeep-19.mat", help="Path to VGG model.")
    parser.add_argument('-c', action="store", dest="c",
                        default="data/original.jpg", help="Path to content image")
    parser.add_argument('-s', action="store", dest="s",
                        default="data/style.jpg", help="Path to style image")
    parser.add_argument('-g', action="store", dest="g",
                        default="generated.jpg", help="Name of generated image.")
    arguments = parser.parse_args()

    layer_weights = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]

    nst = NST(arguments.m, layer_weights, arguments.t, arguments.a, arguments.b)
    optimizer = tf.train.AdamOptimizer(arguments.lr)
    nst.transfer(optimizer, arguments.c, arguments.s, arguments.g, num_epochs=arguments.e)


if __name__ == "__main__":
    main()
    print()
