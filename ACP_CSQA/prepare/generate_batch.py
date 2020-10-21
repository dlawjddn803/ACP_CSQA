import sys

TRAIN_NUM_BATCHES = int(sys.argv[1])
DEV_NUM_BATCHES = int(sys.argv[2])
TEST_NUM_BATCHES = int(sys.argv[3])

train_dataset = "./amr_data/amr_2.0/csqa/train.pred.txt"
dev_dataset = "./amr_data/amr_2.0/csqa/dev.pred.txt"
test_dataset = "./amr_data/amr_2.0/csqa/test.pred.txt"


def process(filename, NUM_BATCHES = 4):
    NUM_BATCHES -= 1
    sentences = []
    cnt = 0

    with open(filename, 'r') as f:
        tot = 0
        lines = f.read().split("\n")
        for k in lines:
            if k.startswith('# ::id '):
                tot += 1
        print(tot)

        bcnt = 0
        for i in lines:
            if i.startswith('# ::id '):
                bcnt += 1
                if bcnt % (tot//NUM_BATCHES) == 0:
                    cnt += 1
                    with open("%s_%d.txt" % (filename[:filename.index('.txt')], cnt), 'w') as bf:
                        for j in sentences:
                            bf.writelines(j + '\n')
                    print(bcnt)
                    sentences = []


            sentences.append(i)
        cnt += 1
        with open("%s_%d.txt" % (filename[:filename.index('.txt')], cnt), 'w') as bf:
            for j in sentences:
                bf.writelines(j + '\n')


if __name__ == '__main__':
    process(train_dataset, TRAIN_NUM_BATCHES)
    process(dev_dataset, DEV_NUM_BATCHES)
    process(test_dataset, TEST_NUM_BATCHES)