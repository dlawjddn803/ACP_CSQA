from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import tensorflow as tf




def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def read_jsonl(input_file):
    with tf.gfile.Open(input_file, 'r') as f:
        return [json.loads(ln) for ln in f]


def preprocess(lines = None, title = None, test_flag=False):
    with open('./csqa/'+str(title[:title.index('.jsonl')])+'.txt', 'w') as f:
        answer_tempelate = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        for line in lines:
            qid = line['id']
            question = convert_to_unicode(line['question']['stem'])
            question_concept = convert_to_unicode(line['question']['question_concept'])
            answers = [
                convert_to_unicode(choice['text'])

                for choice in sorted(
                    line['question']['choices'],
                    key=lambda c: c['label']
                )
            ]
            if test_flag:
                true_answer = 'A'
                index_need = 0
                answer_key = [answer_key for answer_key, index in answer_tempelate.items() if index == index_need][0]
            else:
                answer_key = convert_to_unicode(line['answerKey'])
                true_answer = answers[answer_tempelate[answer_key]]
            f.writelines('# ::id ' + qid + "\n")
            f.writelines("# ::theme " + question_concept + "\n")
            f.writelines("# ::option "+str(answers)+"\n")
            f.writelines("# ::snt "+question+"\n")
            f.writelines("# ::answer_key " + answer_key + "\n")
            f.writelines("# ::true_answer " + true_answer + "\n")
            f.writelines("# ::save-date Fri Jan, 2020\n")
            f.write("(f / follow-02\n      :manner (i / interest-01))")
            f.writelines('\n\n')


if __name__ == '__main__':

    data_list = [
        'train_rand_split.jsonl',
        'dev_rand_split.jsonl',
        'test_rand_split_no_answers.jsonl'
    ]

    test_flag = False
    for data in data_list:
        if 'test' in str(data):
            test_flag = True
        preprocess(lines=read_jsonl('/home/wjddn803/PycharmProjects/gct_dual/dataset/csqa/'+data), title=data, test_flag=test_flag)
