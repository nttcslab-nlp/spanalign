# -*- coding: utf-8 -*-

"""
created by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
updated on Dec. 22, 2020 by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
"""

import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import SQuAD

def makeQA(raw_lines, qa_id_prefix, sent_ngram):
    qa_list = []

    for num_sents in range(1, sent_ngram+1):
        for begin in range(len(raw_lines) - num_sents + 1):
            qa_id = "{}_{}-{}".format(qa_id_prefix, begin, begin+num_sents-1)
            q_str = "\n".join(raw_lines[begin : begin+num_sents])
            qa = SQuAD.SQuAD_QA(qa_id, q_str)

            qa_list.append(qa)

    return qa_list

def main(config):
    title = "{}_{}".format(config.title, config.raw_file_l1.stem)

    with config.raw_file_l1.open() as ifs:
        raw_lines_l1 = [x.strip().replace("　", " ") for x in ifs.readlines()]
    with config.raw_file_l2.open() as ifs:
        raw_lines_l2 = [x.strip().replace("　", " ") for x in ifs.readlines()]

    context_l1 = "\n".join(raw_lines_l1)
    context_l2 = "\n".join(raw_lines_l2)

    paragraph_l1_to_l2 = SQuAD.SQuADParagrap(context_l2)
    paragraph_l2_to_l1 = SQuAD.SQuADParagrap(context_l1)

    qa_id_prefix_l1_to_l2 = title + "_1_2"
    qa_list_l1_to_l2 = makeQA(raw_lines_l1, qa_id_prefix_l1_to_l2,
                              config.ngram)
    qa_id_prefix_l2_to_l1 = title + "_2_1"
    qa_list_l2_to_l1 = makeQA(raw_lines_l2, qa_id_prefix_l2_to_l1,
                              config.ngram)

    paragraph_l1_to_l2.extendQA(qa_list_l1_to_l2)
    paragraph_l2_to_l1.extendQA(qa_list_l2_to_l1)

    squad_data = SQuAD.SQuADDataItem(title)
    squad_data.appendParagraph(paragraph_l1_to_l2)
    squad_data.appendParagraph(paragraph_l2_to_l1)

    squad = SQuAD.SQuAD(version2=True)
    squad.appendData(squad_data)

    with config.output.open("w") as ofs:
        json.dump(squad.toJson(), ofs, ensure_ascii=False, indent=2)

def parse_args():
    parser =  ArgumentParser()
    parser.add_argument('raw_file_l1', type=Path)
    parser.add_argument('raw_file_l2', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('-t', '--title', type=str, default='yomiuri')
    parser.add_argument('-n', '--ngram', type=int, default=4)

    return parser.parse_args()

if __name__ == '__main__':
    config = parse_args()
    main(config)
