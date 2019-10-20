#!/usr/bin/env python

import numpy as np
import random
from keras import backend as K ###

def modulateBPSK(x):
    return -2 * x + 1

def addNoise_fixedlengthCode(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w

def addNoise(x, sigma, len_test = None):
    if len_test is None:
        w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
        positives = K.equal(x, 3)
        positives = K.cast(positives, K.floatx())
        noisy = x + w
        noisy = noisy - noisy*positives + 3*positives
        K.print_tensor(noisy)
        return noisy
    else:
        w = np.random.normal(0.0, sigma, x.shape)
        noisy = x + w
        for noisy_test_i in range(0, noisy.shape[0]):
            if len_test[noisy_test_i][0] < noisy.shape[1]:
                noisy[noisy_test_i][int(len_test[noisy_test_i][0]):] = [3] * (noisy.shape[1] - int(len_test[noisy_test_i][0]))
        return noisy;

def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))


def return_output_shape(input_shape):
    return input_shape


def log_likelihood_ratio(x, sigma):
    return 2 * x / np.float32(sigma ** 2)


def errors(y_true, y_pred):
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)), dtype='float'))


def half_adder(a, b):
    s = a ^ b
    c = a & b
    return s, c


def full_adder(a, b, c):
    s = (a ^ b) ^ c  # for current bit position
    c = (a & b) | (c & (a ^ b))  # for the next bit position
    # print("s: ", s," c: ", c);
    return s, c


def add_bool(a, b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k, dtype=bool)
    c = False
    for i in reversed(range(0, k)):
        s[i], c = full_adder(a[i], b[i], c)
    if c:
        warnings.warn("Addition overflow!")
    return s


def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k - 1, dtype=bool), np.ones(1, dtype=bool)))
    # print("a: ", a," increment: ", increment);
    a = add_bool(a, increment)
    return a


def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0, len(x)):
        x[i] = int('{:0{n}b}'.format(x[i], n=n)[::-1], 2)
    return x


def int2bin(x, N):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        binary = np.zeros((len(x), N), dtype='bool')
        for i in range(0, len(x)):
            binary[i] = np.array([int(j) for j in bin(x[i])[2:].zfill(N)])
    else:
        binary = np.array([int(j) for j in bin(x)[2:].zfill(N)], dtype=bool)

    return binary


def bin2int(b):
    if isinstance(b[0], list):
        integer = np.zeros((len(b),), dtype=int)
        for i in range(0, len(b)):
            out = 0
            for bit in b[i]:
                out = (out << 1) | bit
            integer[i] = out
    elif isinstance(b, np.ndarray):
        if len(b.shape) == 1:
            out = 0
            for bit in b:
                out = (out << 1) | bit
            integer = out
        else:
            integer = np.zeros((b.shape[0],), dtype=int)
            for i in range(0, b.shape[0]):
                out = 0
                for bit in b[i]:
                    out = (out << 1) | bit
                integer[i] = out

    return integer


def polar_design_awgn(N, k, design_snr_dB):
    S = 10 ** (design_snr_dB / 10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))

    A = np.zeros(N, dtype=bool)
    A[idx] = True

    return A


def polar_transform_iter(u):
    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):
        i = 0
        while i < N:
            for j in range(0, n):
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]
            i = i + 2 * n
        n = 2 * n
    return x

def error_correction_hard(clen, received, codebook_decode_array_shuffle = None):
    if codebook_decode_array_shuffle.size != 0:
        codebook = codebook_decode_array_shuffle
    else:
        codebook = code_word_4b6b

    min_hamming_distance = clen + 1
    for key in codebook:
        hamming_distance = 0
        for bit in range(0, clen):
            if received[bit] != key[bit]:
                hamming_distance += 1
        if hamming_distance < min_hamming_distance:
            min_hamming_distance = hamming_distance
            corrected = key
    return corrected

def error_correction_soft(clen, received, codebook_decode_array_shuffle = None):
    if codebook_decode_array_shuffle.size != 0:
        codebook = codebook_decode_array_shuffle
    else:
        codebook = code_word_4b6b

    min_distance = 10.0 ** 10.0
    for key in codebook:
        distance = 0.0
        for bit in range(0, clen):
            distance += abs(received[bit] - (-2.0 * key[bit] + 1.0)) * abs(received[bit] - (-2.0 * key[bit] + 1.0))
        if distance < min_distance:
            # print(distance,"\n")
            min_distance = distance
            corrected = key
    return corrected

def error_correction_soft_DCfreeN5(clen, received):
    if clen == 2:
        codebook = code_word_DCfreeN5_len2
    elif clen == 4:
        codebook = code_word_DCfreeN5_len4
    else:
        print("received word not recoginzed (length can only be 2 or 4)")
        exit(-1)

    min_distance = 10.0 ** 10.0
    for key in codebook:
        distance = 0.0
        for bit in range(0, clen):
            distance += abs(received[bit] - (-2.0 * key[bit] + 1.0)) * abs(received[bit] - (-2.0 * key[bit] + 1.0))
        if distance < min_distance:
            # print(distance,"\n")
            min_distance = distance
            corrected = key
    return corrected

def bit_err(ber, bits, clen):
    """
    bit error rate vs S/N ratio
    :param ber: ber array [sigma, error, bits]
    :param bits: number of bit
    :param clen: code length
    :return:
    """
    biterr = np.zeros((ber.shape[0], 2))
    biterr[:, 0] = 10 * np.log10(1 / (2.0 * ber[:, 0] * ber[:, 0])) - 10 * np.log10(float(bits) / clen)
    biterr[:, 1] = ber[:, 1] / ber[:, 2]
    return biterr

def score(biterr0, biterr1):
    """
    score to evaluate the decoder
    :param biterr0: bit error rate (optimal) [sigma, biterr]
    :param biterr1: bit error rate for evaluation [sigma. biterr]
    :return:
    """

    n = biterr1[0:len(biterr0) - 1, 1]/biterr0[0:len(biterr0) - 1, 1]
    s = np.nansum(n)
    if biterr1[len(biterr0) - 1, 1] == 0:
        s += 1
    else:
        s += 10
    s = s / 5.0
    return s

def scoreBLER(BLER0, BLER1):
    s= 0.0
    for i in range(0, len(BLER0)):
        if BLER1[i] == 0:
            if BLER0[i] == 0:
                s += 0
            else:
                s += 10
        else:
            s += float(BLER0[i]) / float(BLER1[i])

    s = s / len(BLER0)
    return s

def shuffle_code_book(encode_book):
    """
    shuffle the code book
    :param encode_book: code book
    :return: shuffled code book
    """
    codbok = np.array(list(encode_book.items()))
    ids0 = np.random.permutation(codbok.shape[0])
    ids1 = np.random.permutation(codbok.shape[0])

    cod = codbok[ids0, 0]
    word = codbok[ids1, 1]
    shuff_encode_book = dict()

    for i in range(len(cod)):
        shuff_encode_book[cod[i]] = word[i]
    return shuff_encode_book

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def create_codebook_shuffle(nframe = 5):
    cbs = []
    np.random.seed(0)
    for i in range(nframe):
        cbi = shuffle_code_book(codebook_4b6b)
        cbs.append(cbi)

    comb_codbok = combine_codes(cbs)
    return comb_codbok

def combine_codes(codboks):
    """
    combine multiple code books to a big code
    :param codboks: tuple/list of code books
    :return: code book for the combined code
    """
    idx = ()
    for cb in codboks:
        key = cb.keys()
        idx = idx + (list(key),)
    idx = cartesian(idx)

    res = dict()

    cur_index = 0;
    for id in idx:
        cur_index += 1;
        print("processing ", cur_index, " hash entry in the shuffled codebook")

        cod = ''
        word = ''
        for i in range(len(id)):
            cod += id[i]
            word += " " + codboks[i][id[i]]
        cod = np.array(cod.replace('[', ' ').replace(']', ' ').split()).astype(int)
        word = word.lstrip(" ")

        res[str(cod)] = word

    return res

def get_decode_book(codbok):
    """
    get decode book from code boook
    :param codbok: code book
    :return: decode book
    """
    decodbok = dict()
    decodbok_array = []
    for key, val in codbok.items():
        decodbok[str(np.array(val.split()).astype(int))] = key.replace('[', '').replace(']', '')
        decodbok_array.append(np.array(list(val.split(' ')), dtype = 'int'))
    decodbok_array = np.array(decodbok_array)
    return decodbok, decodbok_array


# hash table for encoding
codebook_4b6b = {str(np.array([0, 0, 0, 0])):'0 0 1 1 1 0',
                 str(np.array([0, 0, 0, 1])):'0 0 1 1 0 1',
                 str(np.array([0, 0, 1, 0])):'0 1 0 0 1 1',
                 str(np.array([0, 0, 1, 1])):'0 1 0 1 1 0',
                 str(np.array([0, 1, 0, 0])):'0 1 0 1 0 1',
                 str(np.array([0, 1, 0, 1])):'1 0 0 0 1 1',
                 str(np.array([0, 1, 1, 0])):'1 0 0 1 1 0',
                 str(np.array([0, 1, 1, 1])):'1 0 0 1 0 1',
                 str(np.array([1, 0, 0, 0])):'0 1 1 0 0 1',
                 str(np.array([1, 0, 0, 1])):'0 1 1 0 1 0',
                 str(np.array([1, 0, 1, 0])):'0 1 1 1 0 0',
                 str(np.array([1, 0, 1, 1])):'1 1 0 0 0 1',
                 str(np.array([1, 1, 0, 0])):'1 1 0 0 1 0',
                 str(np.array([1, 1, 0, 1])):'1 0 1 0 0 1',
                 str(np.array([1, 1, 1, 0])):'1 0 1 0 1 0',
                 str(np.array([1, 1, 1, 1])):'1 0 1 1 0 0'}

# hash table for decoding
decode_4b6b = {str(np.array([0, 0, 1, 1, 1, 0])): '0 0 0 0',
               str(np.array([0, 0, 1, 1, 0, 1])):'0 0 0 1',
               str(np.array([0, 1, 0, 0, 1, 1])):'0 0 1 0',
               str(np.array([0, 1, 0, 1, 1, 0])):'0 0 1 1',
               str(np.array([0, 1, 0, 1, 0, 1])):'0 1 0 0',
               str(np.array([1, 0, 0, 0, 1, 1])):'0 1 0 1',
               str(np.array([1, 0, 0, 1, 1, 0])):'0 1 1 0',
               str(np.array([1, 0, 0, 1, 0, 1])):'0 1 1 1',
               str(np.array([0, 1, 1, 0, 0, 1])):'1 0 0 0',
               str(np.array([0, 1, 1, 0, 1, 0])):'1 0 0 1',
               str(np.array([0, 1, 1, 1, 0, 0])):'1 0 1 0',
               str(np.array([1, 1, 0, 0, 0, 1])):'1 0 1 1',
               str(np.array([1, 1, 0, 0, 1, 0])):'1 1 0 0',
               str(np.array([1, 0, 1, 0, 0, 1])):'1 1 0 1',
               str(np.array([1, 0, 1, 0, 1, 0])):'1 1 1 0',
               str(np.array([1, 0, 1, 1, 0, 0])):'1 1 1 1'}

# hash table for error correction during decoding, same as codebook_decode but different type,
# to be compatible with function error_correction_hard() and error_correction_soft()
code_word_4b6b = np.array([[0, 0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 0, 1],
                           [0, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 1, 0],
                           [0, 1, 0, 1, 0, 1],
                           [1, 0, 0, 0, 1, 1],
                           [1, 0, 0, 1, 1, 0],
                           [1, 0, 0, 1, 0, 1],
                           [0, 1, 1, 0, 0, 1],
                           [0, 1, 1, 0, 1, 0],
                           [0, 1, 1, 1, 0, 0],
                           [1, 1, 0, 0, 0, 1],
                           [1, 1, 0, 0, 1, 0],
                           [1, 0, 1, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0],
                           [1, 0, 1, 1, 0, 0]])


vary_length_1_3dk = {
    str(np.array([0])): '0 1',
    str(np.array([1, 0])): '0 0 1',
    str(np.array([1, 1])): '0 0 0 1'
}

vary_length_DCfreeN5_state_1 = {
    str(np.array([0, 1, 0])): [0, 1, 1, 1],
    str(np.array([0, 1, 1])): [0, 1, 0, 1],
    str(np.array([1, 0, 0])): [0, 1, 1, 0],
    str(np.array([1, 0, 1])): [1, 0, 1, 1],
    str(np.array([1, 1, 0])): [1, 0, 0, 1],
    str(np.array([1, 1, 1])): [1, 0, 1, 0],
}

vary_length_DCfreeN5_state_2 = {
    str(np.array([0, 1, 0])): [1, 0, 0, 0],
    str(np.array([0, 1, 1])): [0, 1, 0, 1],
    str(np.array([1, 0, 0])): [0, 1, 1, 0],
    str(np.array([1, 0, 1])): [0, 1, 0, 0],
    str(np.array([1, 1, 0])): [1, 0, 0, 1],
    str(np.array([1, 1, 1])): [1, 0, 1, 0],
}

vary_length_DCfreeN5_for_decode = {
    str(np.array([1, 1])): [0, 0],
    str(np.array([0, 0])): [0, 0],
    str(np.array([0, 1, 1, 1])): [0, 1, 0],
    str(np.array([1, 0, 0, 0])): [0, 1, 0],
    str(np.array([0, 1, 0, 1])): [0, 1, 1],
    str(np.array([0, 1, 1, 0])): [1, 0, 0],
    str(np.array([1, 0, 1, 1])): [1, 0, 1],
    str(np.array([0, 1, 0, 0])): [1, 0, 1],
    str(np.array([1, 0, 0, 1])): [1, 1, 0],
    str(np.array([1, 0, 1, 0])): [1, 1, 1],
}

code_word_DCfreeN5_len2 = np.array([[1, 1], [0, 0]])


code_word_DCfreeN5_len4 = np.array([[0, 1, 1, 1],
                           [1, 0, 0, 0],
                           [0, 1, 0, 1],
                           [0, 1, 1, 0],
                           [1, 0, 1, 1],
                           [0, 1, 0, 0],
                           [1, 0, 0, 1],
                           [1, 0, 1, 0],])

if __name__ == "__main__":
    shuf_code = shuffle_code_book(codebook_4b6b)
    shuf_decode = get_decode_book(shuf_code)
    res = combine_codes([codebook_4b6b, shuf_code])