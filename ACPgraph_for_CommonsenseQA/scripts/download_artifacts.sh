#!/usr/bin/env bash

set -e

 mkdir -p glove
 curl -L -o glove/glove.840B.300d.zip \
     http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip

mkdir -p tools
git clone https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced.git tools/amr-evaluation-tool-enhanced

mkdir -p amr_data
curl -o amr_data/amr_2.0_utils.tar.gz https://www.cs.jhu.edu/~s.zhang/data/AMR/amr_2.0_utils.tar.gz
pushd amr_data
tar -xzvf amr_2.0_utils.tar.gz
rm amr_2.0_utils.tar.gz amr_1.0_utils.tar.gz
popd

