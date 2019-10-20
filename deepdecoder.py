# authors: Congzhe Cao, email:congzhe@ualberta.ca
#          Duanshun Li, email:duanshun@ualberta.ca
# This is the code repo for the paper "Deep-learning based decoding of constrained sequence codes",
# in IEEE Journal on Selected Areas in Communications, https://ieeexplore.ieee.org/document/8792188.
# Credit is also given to Tobias Gruber et al and their github repo https://github.com/gruberto/DL-ChannelDecoding,
# where this code repo is initially partly written based on theirs.
#!/usr/bin/env python

from decoder import *
import draw
import shutil

# force using cpu
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DeepDecoder(Decoder):
    def __init__(self, bits=4, clen=6, frame =1, epoches = 10000, batch_size = 256, code='4b6b'):
        """
        basic settings of the decoder
        :param bits: number of information bits, 4 for the 4b6b code
        :param clen: code length, 6 for the 4b6b code
        :param frame: number of frames
        :param epoches: number of training epoches
        :param batch_size: batch size
        :param code: '4b6b' or '4b6b_shuffle5'
        Note: if train and test for the shuffled concatenated codebook, then always set frame = 1, and adjust bits and clen.
        For example, with 5 4b6b codebooks concatenated and shuffled codebooks, the parameters should be bits=20, clen=30, frame=1,
        and code='4b6b_shuffle5'
        """
        Decoder.__init__(self,bits, clen, frame, code)

        self.epochs = epoches  # number of learning epochs
        self.batch_size = batch_size  # size of batches for calculation the gradient
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'  # 'mse' or 'binary_crossentropy'

    def gen_model_head(self):
        """
        generate head layers for data preparation
        """
        self.model = Sequential()
        self.decoder = Sequential()
        # Define modulator
        self.model.add(
            Lambda(modulateBPSK, input_shape=(self.clen_total,), output_shape=return_output_shape, name="modulator"))
        # Define noise
        self.model.add(
            Lambda(addNoise_fixedlengthCode, arguments={'sigma': self.train_sigma}, output_shape=return_output_shape, name="noise"))
        # Define LLR
        if self.LLR:
            self.model.add(
                Lambda(log_likelihood_ratio, arguments={'sigma': self.train_sigma}, output_shape=return_output_shape, name="LLR"))

    def gen_model_fc(self):
        """
        generate mlp network with all fc layers
        """
        self.gen_model_head()
        # Define decoder
        self.decoder.add(Dense(self.design[0], input_shape=(self.clen_total,), activation='relu', name="decoder00"))
        for i in range(1, len(self.design)):
            self.decoder.add(Dense(self.design[i], activation='relu', name="decoder{:02d}".format(i)))
        self.decoder.add(Dense(self.bits_total, activation='sigmoid', name="prediction"))
        # compile decoder
        self.decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[errors])
        for layer in self.decoder.layers:
            self.model.add(layer)
        # compiler the training model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[ber])

    def gen_model_cnn(self):
        """
        generate cnn network
        :return:
        """
        self.gen_model_head()
        # Define decoder
        self.decoder.add(Reshape((self.clen_total, 1), input_shape=(self.clen_total,)))
        self.decoder.add(Conv1D(self.design[0], self.kernel[0], padding='valid', activation='relu', name="decoder00"))
        for i in range(1, len(self.design)):
            self.decoder.add(Conv1D(self.design[i], self.kernel[i], padding='same', activation='relu', name="decoder{:02d}".format(i)))

        self.decoder.add(Reshape((1, -1), name="FC"))
        self.decoder.add(Dense(self.bits_total, activation='sigmoid'))
        self.decoder.add(Reshape((self.bits_total,), name="pred"))

        self.decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[errors])

        for layer in self.decoder.layers:
            self.model.add(layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[ber])

    def test_NN(self, modelfile=None, opt_ber_file=None, nshuffle=5):
        """
        test the neural networks
        :return:
        """
        if modelfile:
            self.decoder = load_model(modelfile)

        if self.code == "4b6b_shuffle5":
            comb_codbok = create_codebook_shuffle(nframe=nshuffle)

        np.random.seed(0)
        for i in range(0, len(self.sigmas)):
            for ii in range(0, np.round(self.test_words / self.test_batch).astype(int)):
                # Source
                if self.code == "4b6b":
                    x_test, d_test = self.create_words_4b6b(nsample=self.test_batch)
                elif self.code == "4b6b_shuffle5":
                    x_test, d_test = self.create_words_4b6b_shuffle(comb_codbok, nshuffle, nsample=self.test_batch)

                # Modulator (BPSK)
                s_test = -2 * x_test + 1

                # Channel (AWGN)
                y_test = s_test + self.sigmas[i] * np.random.standard_normal(s_test.shape)

                if self.LLR:
                    y_test = 2 * y_test / (self.sigmas[i] ** 2)

                # Decoder
                prob = self.decoder.predict(y_test, verbose=0)
                if self.probs is not None:
                    self.test_x = np.vstack((self.test_x, y_test))
                    self.test_d = np.vstack((self.test_d, d_test))
                    self.probs = np.vstack((self.probs, prob))
                else:
                    self.test_x = y_test
                    self.test_d = d_test
                    self.probs = prob

                self.nb_errors[i] += self.decoder.evaluate(y_test, d_test, batch_size=self.test_batch, verbose=0)[1]
                self.nb_bits[i] += d_test.size

        biterr_i = bit_err(np.array([self.sigmas, self.nb_errors, self.nb_bits]).T, self.bits, self.clen)

        s = 0
        if opt_ber_file:
            ber0 = np.loadtxt(opt_ber_file)
            biterr0 = bit_err(ber0, self.bits, self.clen)
            s = score(biterr0, biterr_i)

        print("score: {}".format(s))
        return s


def train(args):
    dec = DeepDecoder(4, 6, 1, 10000, 256, code = '4b6b')
    if (args.which == "MLP"):
        dec.design = [64, 32, 16]
        dec.gen_model_fc()
    elif (args.which == "CNN"):
        dec.design = [8, 14, 8]
        dec.kernel = [3, 3, 3]
        dec.gen_model_cnn()
    else:
        print("model needs to be selected: MLP or CNN")
        return;

    # set number of tests
    dec.test_setup(0, 10, 5, 1000, 1000)

    if not os.path.isdir('./tmpfix'):
        os.mkdir('./tmpfix')
    modelfile = "./tmpfix/4b6b-f{}-".format(dec.frame) + ("-".join(str(n) for n in dec.design)) + "-" + args.which + ".h5"
    dec.train(modelfile, interval=5000, saveber=True, topk=2, record_from=500, nshuffle=2)
    draw.plot_test_from_files("./tmpfix", './compare-f{}.png'.format(dec.frame), dec.bits, dec.clen)


def test(args):
    dec = DeepDecoder(4, 6, 1)
    if (args.which == "MLP"):
        dec.design = [64, 32, 16]
        dec.gen_model_fc()
    elif (args.which == "CNN"):
        dec.design = [8, 14, 8]
        dec.kernel = [3, 3, 3]
        dec.gen_model_cnn()

    dec.decoder.load_weights(args.model)
    dec.test_setup(0, 10, 5, 1000, 1000)
    dec.test_NN(opt_ber_file="look-up-f1-soft.ber")
    dec.save_test_result(args.model.replace('.h5','-test.ber'), None)
    draw.plot_test_result(dec, args.model.replace('.h5','-test.png'))
    draw.plot_test_from_files("./tmpfix", './compare-f{}.png'.format(dec.frame), dec.bits, dec.clen)

def testshuffle(args):
    dec = DeepDecoder(20, 30, 1, code = '4b6b_shuffle5')
    dec.decoder = load_model(args.model, custom_objects={'errors': errors})
    dec.test_setup(0, 10, 5, 1000, 1000)
    dec.test_NN(opt_ber_file="look-up-f1-soft-True.ber", nshuffle=2)
    dec.save_test_result(args.model.replace('.h5', '-test-shuffle.ber'), None)
    draw.plot_test_result(dec, args.model.replace('.h5', '-test-shuffle.png'))
    draw.plot_test_from_files("./tmpfix", './compare-shuflle.png', dec.bits, dec.clen)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert data to lmdb format.')
    parser.add_argument('what',
                        metavar="train or test (default train and test)",
                        help='train or test')
    parser.add_argument('--which', '-w',
                        metavar="MLP or CNN",
                        help='MLP or CNN')
    parser.add_argument('--model', '-m',
                        metavar="/path/to/model",
                        help='/path/to/model')
    parser.add_argument('--shuffle', '-s',
                        action='store_true',
                        default=False,
                        help='test shuffle')

    args = parser.parse_args()

    if args.what == "train":
        train(args)
    elif args.what == "test":
        if args.shuffle:
            testshuffle(args)
        else:
            test(args)

