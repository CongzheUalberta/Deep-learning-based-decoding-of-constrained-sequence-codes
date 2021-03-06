# Deep-learning-based-decoding-of-constrained-sequence-codes
============================

This repo contains the code and data for [Deep-learning-based-decoding-of-constrained-sequence-codes](https://ieeexplore.ieee.org/abstract/document/8792188) by [Congzhe Cao](https://scholar.google.ca/citations?user=lWZ14igAAAAJ&hl=en&oi=ao), [Duanshun Li](https://scholar.google.ca/citations?hl=en&user=RbDI3VUAAAAJ), and [Ivan Fair](http://www.ece.ualberta.ca/~fair/).

Credit is also given to Tobias Gruber et al and their github repo https://github.com/gruberto/DL-ChannelDecoding, where this code repo is initially partly written based on theirs.

## Abstract
Constrained sequence (CS) codes, including fixedlength CS codes and variable-length CS codes, have been widely used in modern wireless communication and data storage systems. Sequences encoded with constrained sequence codes satisfy constraints imposed by the physical channel to enable efficient and reliable transmission of coded symbols. In this paper, we propose using deep learning approaches to decode fixed-length and variable-length CS codes. Traditional encoding and decoding of fixed-length CS codes rely on look-up tables (LUTs), which is prone to errors that occur during transmission. We introduce fixed-length constrained sequence decoding based on multiple layer perception (MLP) networks and convolutional neural networks (CNNs), and demonstrate that we are able to achieve low bit error rates that are close to maximum a posteriori probability (MAP) decoding as well as improve the system throughput. Further, implementation of capacity-achieving fixedlength codes, where the complexity is prohibitively high with LUT decoding, becomes practical with deep learning-based decoding. We then consider CNN-aided decoding of variable-length CS codes. Different from conventional decoding where the received sequence is processed bit-by-bit, we propose using CNNs to perform one-shot batch-processing of variable-length CS codes such that an entire batch is decoded at once, which improves the system throughput. Moreover, since the CNNs can exploit global information with batch-processing instead of only making use of local information as in conventional bit-by-bit processing, the error rates can be reduced. We present simulation results that show excellent performance with both fixed-length and variable-length CS codes that are used in the frontiers of wireless communication systems.

============================

```
C. Cao, D. Li and I. Fair, "Deep Learning-Based Decoding of Constrained Sequence Codes," in IEEE Journal on Selected Areas in Communications.
```
