#!/bin/sh

mkdir images && \
mkdir images/train && \
mkdir images/test && \
\
(cd images/train && wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && gunzip *.gz) && \
(cd images/test  && wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && gunzip *.gz)

