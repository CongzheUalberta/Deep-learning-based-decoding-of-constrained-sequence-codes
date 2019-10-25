# authors: Congzhe Cao, email:congzhe@ualberta.ca
#          Duanshun Li, email:duanshun@ualberta.ca
# This is the code repo for the paper "Deep-learning based decoding of constrained sequence codes",
# in IEEE Journal on Selected Areas in Communications, https://ieeexplore.ieee.org/document/8792188.
# Credit is also given to Tobias Gruber et al and their github repo https://github.com/gruberto/DL-ChannelDecoding,
# where this code repo is initially partly written based on theirs.
#!/usr/bin/env python

#!/usr/bin/env python
import os, glob
import numpy as np
import matplotlib.pyplot as plt


def plot_test_result(decoder, testfile=None, BLER=None):
    """
    plot result and save to file
    :param picfile: image file
    :return:
    """
    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111)

    legend = []
    ax.plot(10 * np.log10(1 / (2.0 * decoder.sigmas ** 2)) - 10 * np.log10(float(decoder.bits) / decoder.clen),
             decoder.nb_errors.astype(float) / decoder.nb_bits, label='NN')

    legend.append('NN')
    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BER')
    plt.grid(True)

    if BLER: # for variable-length coding
        ax.plot(10 * np.log10(1 / (2.0 * decoder.sigmas ** 2)) - 10 * np.log10(float(decoder.bits) / decoder.clen),
                 BLER / decoder.nb_bits, label='BLER')

        legend.append('raw BLER')
        plt.legend(legend, loc=3)
        plt.yscale('log')
        plt.xlabel('$E_b/N_0$')
        plt.ylabel('BLER')

    if testfile:
        plt.savefig(testfile)
        plt.close()
    else:
        plt.show()




def plot_test_from_file(datafile, picfile, bits, clen):
    """
    read from file and draw figure
    :param datafile: input file of the test result
    :param picfile: output .png file
    :param bits: information bits per frame
    :param clen: code length per frame
    :return:
    """
    # read from file
    data = np.loadtxt(datafile)
    sigmas = data[:, 0]
    nb_errors = data[:, 1]
    nb_bits = data[:, 2]

    # plot
    legend = []
    plt.plot(10 * np.log10(1 / (2.0 * sigmas*sigmas)) - 10 * np.log10(float(bits) / clen),
             nb_errors.astype(float) / nb_bits)
    legend.append(os.path.splitext(picfile)[0])
    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BER')
    plt.grid(True)
    plt.savefig(picfile)
    plt.close()

def plot_test_from_files(datadir, picfile, bits, clen):
    """
    read from files and draw figure
    :param datadir: folder for .ber files
    :param picfile: output .png file
    :param bits: information bits
    :param clen: code length
    :return:
    """
    files = glob.glob("*.ber")
    #os.chdir(datadir)
    files += glob.glob(datadir+"/*.ber")
    fig =  plt.figure(figsize=(6, 5))
    markers = ['P', '*', '>', 'd', '<', 's']
    legend = []
    id = 0
    for f in files:
        data = np.loadtxt(f)
        sigmas = data[:, 0]
        nb_errors = data[:, 1]
        nb_bits = data[:, 2]
        legend = f
        plt.plot(10 * np.log10(1 / (2.0 * sigmas * sigmas)) - 10 * np.log10(float(bits) / clen),
                 nb_errors.astype(float) / nb_bits, label=legend, lw=1, marker=markers[id], markersize=4)
        id += 1

    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BLER')
    plt.grid(True)
    plt.legend()
    plt.savefig(picfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_log(log, savefile=None):
    """
    plot training history
    :param log: the training log such as history.history['loss']
    :param savefile: file to save the figure, if None show the figure
    :return:
    """
    plt.yscale('log')
    plt.plot(log)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid(True)
    if savefile:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()


def plot_test_score(scores, savefile=None):
    plt.figure(figsize=(16, 9))
    plt.plot(scores[:, 0], scores[:, 1])
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.grid(True)
    if savefile:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    plot_test_from_files("./drawComparison/", './compare-f{}.png'.format(3), 4, 6)





