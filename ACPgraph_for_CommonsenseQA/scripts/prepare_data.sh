#!/bin/bash

usage() {
    echo "Usage: $0 -v <AMR corpus version. Possible values: 1 or 2> -p <Path to AMR corpus>"
    echo "  Make sure your AMR corpus is untouched."
    echo "  It should organized like below:"
    echo "  <AMR corpus>"
    echo "      data/"
    echo "      docs/"
    echo "      index.html"
    exit 1;
}

while getopts ":h:v:p:" o; do
    case "${o}" in
        h)
            usage
            ;;
        v)
            v=${OPTARG}
            ((v == 1 || v == 2)) || usage
            ;;
        p)
            p=${OPTARG}
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z $v ]; then
    usage
fi

if [ -z $p ]; then
    usage
fi


if [[ "$v" == "2" ]]; then
    DATA_DIR=$p/amr_data/amr_2.0/csqa
    SPLIT_DIR=$p/dataset/csqa
    TRAIN=${SPLIT_DIR}/train
    DEV=${SPLIT_DIR}/dev
    TEST=${SPLIT_DIR}/test
else
    DATA_DIR=$p/amr_data/amr_2.0/csqa
    SPLIT_DIR=$p/dataset/csqa
    TRAIN=${SPLIT_DIR}/train
    DEV=${SPLIT_DIR}/dev
    TEST=${SPLIT_DIR}/test
fi

echo "Preparing data in ${DATA_DIR}...`date`"
mkdir -p ${DATA_DIR}
awk FNR!=0 ${TRAIN}/* > ${DATA_DIR}/train.txt
awk FNR!=0 ${DEV}/* > ${DATA_DIR}/dev.txt
awk FNR!=0 ${TEST}/* > ${DATA_DIR}/test.txt
echo "Done..`date`"

