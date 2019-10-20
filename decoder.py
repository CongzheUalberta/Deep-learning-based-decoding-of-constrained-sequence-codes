#!/usr/bin/env python

from utils import *

from keras.models import Sequential
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers import Conv1D, BatchNormalization
from keras.models import load_model
from keras.layers import MaxPooling1D
from keras.utils.np_utils import to_categorical

import draw
import os


class Decoder:
    def __init__(self, bits=4, clen=6, frame=1, code='4b6b'):
        """
        basic settings of the decoder
        :param bits: number of information bits
        :param clen: code length
        :param frame: number of frames in one-shot decoding
        """
        self.bits = bits  # number of information bits
        self.clen = clen  # code length
        self.frame = frame;  # number of frames in one-shot decoding
        self.bits_total = self.bits * frame;  # total information bits in all frames
        self.clen_total = self.clen * frame;  # total code length in all frames
        self.LLR = False  # 'True' enables the log-likelihood-ratio layer

        self.train_SNR_Eb = 1.0  # training-Eb/No
        self.code = code  # type of code ('random' or 'polar')
        self.train_SNR_Es = self.train_SNR_Eb + 10 * np.log10(float(self.bits) / self.clen)
        self.train_sigma = np.sqrt(1/(2*10**(self.train_SNR_Es/10))) # original code

        # for test
        self.test_x = None  # code word
        self.test_d = None  # code
        self.probs = None  # probabilities

        # models
        self.model = None  # model for training
        self.decoder = None  # model for decoding

    def gen_train_data(self, nshuffle=5):

        if self.code == "4b6b":
            x, lb = self.create_words_4b6b()
        elif self.code == '1_3dk':
            x, lb, _ = self.create_words_1_3dk()
        elif self.code == '1_3dk_with_padding':
            x, lb, _, len_test = self.create_words_1_3dk_with_padding()
        elif self.code == 'DCfreeN5':
            x, lb, _ = self.create_words_DCfreeN5()
        elif self.code == 'DCfreeN5_with_padding':
            x, lb, _, len_test = self.create_words_DCfreeN5_with_padding()
        elif self.code == "4b6b_shuffle5":
            comb_codbok = create_codebook_shuffle(nframe=nshuffle)
            x, lb = self.create_words_4b6b_shuffle(comb_codbok, n=nshuffle)
        else:
            raise ValueError('Cannot recognize the code type.')

        self.x_train = x # codewords
        self.lb_train = lb # labels

    def train(self, modelfile, interval=10000, saveber=False, topk=3, record_from=10000, nshuffle=5, isContinue=False, fromEpoch=None):
        """
        :param interval: test interval
        :param saveber: save ber file or not
        :param topk: keep the top 3 best
        :param record_from: start testing from record_from
        :param nshuffle: number of shuffles for shuffled code book
        :return:
        """
        self.model.summary()

        self.gen_train_data(nshuffle=nshuffle)
        print("Data generated, start training!")

        epoch = record_from
        history = self.model.fit(self.x_train, self.lb_train, batch_size=self.batch_size, epochs=epoch, verbose=1,
                                 shuffle=True)
        if isContinue:
            epoch = fromEpoch

        test_res = []
        while epoch < self.epochs:
            history_i = self.model.fit(self.x_train, self.lb_train, batch_size=self.batch_size, epochs=interval,
                                      verbose=1, shuffle=True)
            epoch += interval
            if self.code == '4b6b' or self.code == '4b6b_shuffle5':
                test_me = self.test_NN(opt_ber_file="look-up-f1-soft.ber", nshuffle=nshuffle)
            else: # variable-length decoding
                test_me, BLER = self.test_NN_with_padding(opt_ber_file="look-up-f1-soft.ber", isContinue=isContinue)
            test_res.append([epoch, test_me])

            modelfilei = modelfile.replace('.h5', '_epoch_{:09d}.h5'.format(epoch))

            if saveber:
                if self.code == '4b6b' or self.code == '4b6b_shuffle5':
                    BLER = None
                self.save_test_result(modelfilei.replace('.h5', '-test.ber'), BLER)
                draw.plot_test_result(self, modelfilei.replace('.h5', '-test.png'), BLER=BLER)

            self.decoder.save(modelfilei)

            history.history['loss'] = history.history['loss'] + history_i.history['loss']

        test_res = np.array(test_res)
        np.savetxt(modelfile.replace('.h5', '.test'), np.array(test_res))
        draw.plot_training_log(history.history['loss'], modelfile.replace('.h5', '-train-log.png'))
        draw.plot_test_score(test_res, modelfile.replace('.h5', '-test-score.png'))

        # cleaning
        test_res = test_res[test_res[:, 1].argsort()]
        for i in np.arange(topk, test_res.shape[0], 1):  # keep only top 5
            modelfilei = modelfile.replace('.h5', '_epoch_{:09d}.h5'.format(int(test_res[i][0])))
            os.remove(modelfilei)
            if saveber:
                os.remove(modelfilei.replace('.h5', '-test.ber'))
                os.remove(modelfilei.replace('.h5', '-test.png'))

        self.decoder.save(modelfile)

    def test_setup(self, SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points, num_words=10000, test_batch=1000):
        """
        test settings
        :param num_words: number of word for testing
        :param test_batch: number of word per batch, increase(decrease) the batch with small(large) nets
        :return:
        """
        SNR_dB_start_Es = SNR_dB_start_Eb + 10 * np.log10(float(self.bits) / self.clen)
        SNR_dB_stop_Es = SNR_dB_stop_Eb + 10 * np.log10(float(self.bits) / self.clen)

        self.SNR_dBs = np.linspace(SNR_dB_start_Es, SNR_dB_stop_Es, SNR_points)

        sigma_start = np.sqrt(1 / (2 * 10 ** (SNR_dB_start_Es / 10.0)))
        sigma_stop = np.sqrt(1 / (2 * 10 ** (SNR_dB_stop_Es / 10.0)))

        # for fixed-length decoding
        self.sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

        # for variable-length decoding
        #self.sigmas = np.sqrt(1 / (2 * 10 ** (self.SNR_dBs / 10.0)))

        self.nb_errors = np.zeros(len(self.sigmas), dtype=int)
        self.nb_bits = np.zeros(len(self.sigmas), dtype=int)

        self.test_words = num_words
        self.test_batch = test_batch

    def save_test_result(self, testfile, BLER):
        """
        save test result to file
        :param testfile: txt file
        :return:
        """
        # save to file
        if self.code == "4b6b" or self.code == "4b6b_shuffle5":
            BLER = [0,0,0,0,0]
        data = np.array([self.sigmas, self.nb_errors, self.nb_bits, BLER])
        np.savetxt(testfile, data.T)

        self.probs = None
        if self.probs is not None:
            np.savez(testfile.replace('.ber', '-probs'), x=self.test_x, d=self.test_d, probs=self.probs)

    def create_words_4b6b(self, outfile=None, nsample=None):
        """
        :param nsample: number of samples
        :return:
        """
        if nsample is None:  # Create all possible information words
            nsample = 2 ** self.bits_total
            d = np.zeros((nsample, self.bits_total), dtype=int)
            for i in range(1, nsample):
                d[i] = inc_bool(d[i - 1])
                # print(i," : ",d[i]);
        else:
            d = np.random.randint(0, 2, size=(nsample, self.bits_total))

        x = np.zeros((nsample, self.clen_total), dtype=int)
        for i in range(0, nsample):
            for f in range(0, self.frame):
                tmp = d[i][0 + self.bits * f:self.bits + self.bits * f]
                x[i][0 + self.clen * f:self.clen + self.clen * f] = np.fromstring(codebook_4b6b[str(tmp)], dtype=int,
                                                                                  sep=' ')
        if outfile:
            np.savez_compressed(outfile, x=x, d=d)
        return x, d

    def create_words_4b6b_shuffle(self, comb_codbok, n=5, outfile=None, nsample=None):

        if nsample is None:  # Create all possible information words
            nsample = 2 ** self.bits_total
            d = np.zeros((nsample, self.bits_total), dtype=int)
            for i in range(1, nsample):
                d[i] = inc_bool(d[i - 1])
                # print(i," : ",d[i]);
        else:
            d = np.random.randint(0, 2, size=(nsample, self.bits_total))

        x = np.zeros((nsample, self.clen_total), dtype=int)
        for i in range(0, nsample):
            for f in range(0, self.frame):
                tmp = d[i][0 + self.bits * f:self.bits + self.bits * f]
                x[i][0 + self.clen * f:self.clen + self.clen * f] = np.fromstring(comb_codbok[str(tmp)], dtype=int,
                                                                                  sep=' ')
        if outfile:
            np.savez_compressed(outfile, x=x, d=d)
        return x, d

    def create_words_1_3dk(self, outfile=None, nsample=None):
        if nsample is None:  # Create all possible information words
            nsample = 2 ** self.bits_total
            d = np.zeros((nsample, self.bits_total), dtype=int)
            for i in range(1, nsample):
                d[i] = inc_bool(d[i - 1])
                # print(i," : ",d[i]);
        else:
            d = np.random.randint(0, 2, size=(nsample, self.bits_total))  # increase the test code length

        code_seq_len = 12
        code_seq_label = 6
        lab = np.zeros((d.shape[0], int(code_seq_label)))
        x = np.zeros((d.shape[0], code_seq_len))
        for j in range(d.shape[0]):
            di = d[j]
            xi = []
            lbi = []
            i = 0

            #print("di", di)
            # first do i = 0
            if di[i] == 1:
                if i + 1 >= di.shape[0]:
                    break
                elif di[i + 1] == 0:  # 10 -> 001
                    lbi = [3]
                    xi = xi + [0, 0, 1]
                else:  # 11 -> 0001
                    xi = xi + [0, 0, 0, 1]
                    lbi = [4]
                i += 2
            else:  # 0 -> 01
                lbi = [2]
                xi = xi + [0, 1]
                i += 1

            # then do i > 0
            while len(xi) < code_seq_len:
                if di[i] == 1:
                    if di[i + 1] == 0:  # 10 -> 001
                        lbi = lbi + [lbi[- 1] + 3]
                        xi = xi + [0, 0, 1]
                    else: # 11 -> 0001
                        xi = xi + [0, 0, 0, 1]
                        lbi = lbi + [lbi[- 1] + 4]
                    i += 2
                else:  # 0 -> 01
                    lbi = lbi + [lbi[len(lbi) - 1] + 2]
                    xi = xi + [0, 1]
                    i += 1
            lbi=lbi[:code_seq_label];
            lab[j, :len(lbi)] = lbi
            if lab[j][len(lbi) - 1] > code_seq_len:
                for lab_i in range(len(lbi)-1, code_seq_label):
                    lab[j][lab_i] = code_seq_len + 1
            else:
                if lab[j][len(lbi) - 1] + 3 > code_seq_len:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1
                else:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1


            print("xi", xi)
            print("x[j]", d[j])
            print("lab[j]", lab[j])
            x[j, :(code_seq_len+1)] = xi[:code_seq_len]


        #lab = to_categorical(lab)

        if outfile:
            np.savez_compressed(outfile, x=x, lab=lab, d=d)

        return x, lab, d

    def create_words_1_3dk_with_padding(self, outfile=None, nsample=None):
        if nsample is None:  # Create all possible information words
            nsample = 2 ** self.bits_total
            d = np.zeros((nsample, self.bits_total), dtype=int)
            for i in range(1, nsample):
                d[i] = inc_bool(d[i - 1])
                # print(i," : ",d[i]);
        else:
            d = np.random.randint(0, 2, size=(nsample, self.bits_total))  # increase the test code length

        code_seq_len = 12
        code_seq_label = 6
        lab = np.zeros((d.shape[0], int(code_seq_label)))
        x = np.zeros((d.shape[0], code_seq_len))
        self.len_test = np.zeros((d.shape[0], 1), dtype=int)
        for j in range(d.shape[0]):
            di = d[j]
            xi = []
            lbi = []
            i = 0

            #print("di", di)
            # first do i = 0
            if di[i] == 1:
                if i + 1 >= di.shape[0]:
                    break
                elif di[i + 1] == 0:  # 10 -> 001
                    lbi = [3]
                    xi = xi + [0, 0, 1]
                else:  # 11 -> 0001
                    xi = xi + [0, 0, 0, 1]
                    lbi = [4]
                i += 2
            else:  # 0 -> 01
                lbi = [2]
                xi = xi + [0, 1]
                i += 1

            # then do i > 0
            while i < self.bits_total:
                if di[i] == 1 and i !=5:
                    if di[i + 1] == 0:  # 10 -> 001
                        lbi = lbi + [lbi[- 1] + 3]
                        xi = xi + [0, 0, 1]
                    else: # 11 -> 0001
                        xi = xi + [0, 0, 0, 1]
                        lbi = lbi + [lbi[- 1] + 4]
                    i += 2
                else:  # 0 -> 01
                    lbi = lbi + [lbi[len(lbi) - 1] + 2]
                    xi = xi + [0, 1]
                    i += 1
            lbi=lbi[:code_seq_label];
            lab[j, :len(lbi)] = lbi
            if lab[j][len(lbi) - 1] > code_seq_len:
                for lab_i in range(len(lbi)-1, code_seq_label):
                    lab[j][lab_i] = code_seq_len + 1
            else:
                if lab[j][len(lbi) - 1] + 3 > code_seq_len:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1
                else:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1

            self.len_test[j] = len(xi)
            if len(xi) < code_seq_len:
                for xi_i in range(len(xi), code_seq_len):
                    xi = xi + [-1]
            x[j, :code_seq_len] = xi[:code_seq_len]

            #print("di", d[j])
            #print("xi", xi)
            #print("len_test", self.len_test[j])
            #print("lab[j]\n", lab[j])


        #lab = to_categorical(lab)

        if outfile:
            np.savez_compressed(outfile, x=x, lab=lab, d=d)

        return x, lab, d, self.len_test

    def create_words_DCfreeN5(self, outfile=None, nsample=None):
        if nsample is None:  # Create all possible information words
            nsample = 2 ** self.bits_total
            d = np.zeros((nsample, self.bits_total), dtype=int)
            for i in range(1, nsample):
                d[i] = inc_bool(d[i - 1])
                # print(i," : ",d[i]);
        else:
            d = np.random.randint(0, 2, size=(nsample, self.bits_total))  # increase the test code length

        code_seq_len = 12
        code_seq_label = 6
        lab = np.zeros((d.shape[0], int(code_seq_label)))
        x = np.zeros((d.shape[0], code_seq_len))
        state = np.zeros((d.shape[0], int(code_seq_label)))
        for j in range(d.shape[0]):
            di = d[j]
            xi = []
            lbi = []
            state_i = []
            i = 0

            #print("di", di)
            # first do i = 0
            if di[i] == 0 and di[i + 1] == 0:
                # assume initial state is 1
                lbi = lbi + [2]
                xi = xi + [1, 1]
                state_i = state_i + [2]
                i += 2
            else:
                # source sequence not 00
                lbi = lbi + [4]
                xi = xi + vary_length_DCfreeN5_state_1[str(di[i: i + 3])]
                if np.array_equal(di[i: i + 3], [0, 1, 0]) or np.array_equal(di[i: i + 3], [1, 0, 1]):
                    state_i = state_i + [2]
                else:
                    state_i = state_i + [1]
                i += 3

            # then do i > 0
            while len(xi) < code_seq_len:
                if state_i[len(lbi) - 1] == 1:
                    # if in state 1
                    if di[i] == 0 and di[i + 1] == 0:
                        lbi = lbi + [lbi[-1] + 2]
                        xi = xi + [1, 1]
                        state_i = state_i + [2]
                        i += 2
                    else:
                        # source sequence not 00
                        lbi = lbi + [lbi[-1] + 4]
                        xi = xi + vary_length_DCfreeN5_state_1[str(di[i: i + 3])]
                        if np.array_equal(di[i: i + 3], [0, 1, 0]) or np.array_equal(di[i: i + 3], [1, 0, 1]):
                            state_i = state_i + [2]
                        else:
                            state_i = state_i + [1]
                        i += 3
                else:
                    if di[i] == 0 and di[i + 1] == 0:
                        # if in state 0
                        lbi = lbi + [lbi[-1] + 2]
                        xi = xi + [0, 0]
                        state_i = state_i + [1]
                        i += 2
                    else:
                        # source sequence not 00
                        lbi = lbi + [lbi[-1] + 4]
                        xi = xi + vary_length_DCfreeN5_state_2[str(di[i: i + 3])]
                        if np.array_equal(di[i: i + 3], [0, 1, 0]) or np.array_equal(di[i: i + 3], [1, 0, 1]):
                            state_i = state_i + [1]
                        else:
                            state_i = state_i + [2]
                        i += 3

            lbi=lbi[:code_seq_label];
            lab[j, :len(lbi)] = lbi
            if lab[j][len(lbi) - 1] > code_seq_len:
                for lab_i in range(len(lbi)-1, code_seq_label):
                    lab[j][lab_i] = code_seq_len + 1
            else:
                if lab[j][len(lbi) - 1] + 3 > code_seq_len:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1
                else:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1

            #print("xi", xi)
            #print("d[j]", d[j])
            #print("lab[j]", lab[j])
            x[j, :(code_seq_len+1)] = xi[:code_seq_len]


        #lab = to_categorical(lab)

        if outfile:
            np.savez_compressed(outfile, x=x, lab=lab, d=d)

        return x, lab, d

    def create_words_DCfreeN5_with_padding(self, outfile=None, nsample=None):
        if nsample is None:  # Create all possible information words
            nsample = 2 ** self.bits_total
            d = np.zeros((nsample, self.bits_total), dtype=int)
            for i in range(1, nsample):
                d[i] = inc_bool(d[i - 1])
                # print(i," : ",d[i]);
        else:
            d = np.random.randint(0, 2, size=(nsample, self.bits_total))  # increase the test code length

        code_seq_len = 12
        code_seq_label = 6
        lab = np.zeros((d.shape[0], int(code_seq_label)))
        x = np.zeros((d.shape[0], code_seq_len))
        self.len_test = np.zeros((d.shape[0], 1), dtype=int)
        state = np.zeros((d.shape[0], int(code_seq_label)))
        for j in range(d.shape[0]):
            di = d[j]
            xi = []
            lbi = []
            state_i = []
            i = 0

            #print("di", di)
            # first do i = 0
            if di[i] == 0 and di[i + 1] == 0:
                # assume initial state is 1
                lbi = lbi + [2]
                xi = xi + [1, 1]
                state_i = state_i + [2]
                i += 2
            else:
                # source sequence not 00
                lbi = lbi + [4]
                xi = xi + vary_length_DCfreeN5_state_1[str(di[i: i + 3])]
                if np.array_equal(di[i: i + 3], [0, 1, 0]) or np.array_equal(di[i: i + 3], [1, 0, 1]):
                    state_i = state_i + [2]
                else:
                    state_i = state_i + [1]
                i += 3

            # then do i > 0
            # self.bits_total would be 9 if code_seq_len = 12
            while i + 3 <= self.bits_total:
                if state_i[len(lbi) - 1] == 1:
                    # if in state 1
                    if di[i] == 0 and di[i + 1] == 0:
                        lbi = lbi + [lbi[-1] + 2]
                        xi = xi + [1, 1]
                        state_i = state_i + [2]
                        i += 2
                    else:
                        # source sequence not 00
                        lbi = lbi + [lbi[-1] + 4]
                        xi = xi + vary_length_DCfreeN5_state_1[str(di[i: i + 3])]
                        if np.array_equal(di[i: i + 3], [0, 1, 0]) or np.array_equal(di[i: i + 3], [1, 0, 1]):
                            state_i = state_i + [2]
                        else:
                            state_i = state_i + [1]
                        i += 3
                else:
                    if di[i] == 0 and di[i + 1] == 0:
                        # if in state 0
                        lbi = lbi + [lbi[-1] + 2]
                        xi = xi + [0, 0]
                        state_i = state_i + [1]
                        i += 2
                    else:
                        # source sequence not 00
                        lbi = lbi + [lbi[-1] + 4]
                        xi = xi + vary_length_DCfreeN5_state_2[str(di[i: i + 3])]
                        if np.array_equal(di[i: i + 3], [0, 1, 0]) or np.array_equal(di[i: i + 3], [1, 0, 1]):
                            state_i = state_i + [1]
                        else:
                            state_i = state_i + [2]
                        i += 3
            # deal with tailing source bits
            if i == self.bits_total - 2 and di[i] == 0 and di[i + 1] == 0:
                if state_i[len(lbi) - 1] == 1:
                    lbi = lbi + [lbi[-1] + 2]
                    xi = xi + [1, 1]
                    state_i = state_i + [2]
                    i += 2
                else:
                    lbi = lbi + [lbi[-1] + 2]
                    xi = xi + [0, 0]
                    state_i = state_i + [1]
                    i += 2

            lbi=lbi[:code_seq_label];
            lab[j, :len(lbi)] = lbi
            if lab[j][len(lbi) - 1] > code_seq_len:
                for lab_i in range(len(lbi)-1, code_seq_label):
                    lab[j][lab_i] = code_seq_len + 1
            else:
                if lab[j][len(lbi) - 1] + 3 > code_seq_len:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1
                else:
                    for lab_i in range(len(lbi), code_seq_label):
                        lab[j][lab_i] = code_seq_len + 1

            self.len_test[j] = len(xi)
            if len(xi) < code_seq_len:
                for xi_i in range(len(xi), code_seq_len):
                    xi = xi + [-1]
            x[j, :code_seq_len] = xi[:code_seq_len]

            #print("di", d[j])
            #print("xi", xi)
            #print("len_test", self.len_test[j])
            #print("lab[j]", lab[j], "\n")

        if outfile:
            np.savez_compressed(outfile, x=x, lab=lab, d=d)

        return x, lab, d, self.len_test
