#!/usr/bin/env python3

"""
modified on Dec. 22, 2020 by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
original: thompsonb/vecalign score.py and dp_utils.py
"""

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import argparse
import sys
from collections import defaultdict
from ast import literal_eval

import numpy as np

"""
Faster implementation of lax and strict precision and recall, based on
   https://www.aclweb.org/anthology/W11-4624/.

"""


def read_alignments(fin):
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))

    # I know bluealign files have a few entries entries missing,
    #   but I don't fix them in order to be consistent previous reported scores
    return alignments

def _precision(goldalign, testalign):
    """
    Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter

    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    # mappings from source test sentence idxs to
    #    target gold sentence idxs for which the source test sentence
    #    was found in corresponding source gold alignment
    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)

    for (test_src, test_target) in testalign:
        if (test_src, test_target) == ((), ()):
            continue
        if (test_src, test_target) in goldalign:
            # strict match
            tpstrict += 1
            tplax += 1
        else:
            # For anything with partial gold/test overlap on the source,
            #   see if there is also partial overlap on the gold/test target
            # If so, its a lax match
            target_ids = set()
            for src_test_id in test_src:
                for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                    target_ids.add(tgt_id)
            if set(test_target).intersection(target_ids):
                tplax += 1
            else:
                fplax += 1
            fpstrict += 1

    return np.array([tpstrict, fpstrict, tplax, fplax], dtype=np.int32)


def score_multiple(gold_list, test_list, value_for_div_by_0=0.0, delete=True):
    # accumulate counts for all gold/test files
    pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    for goldalign, testalign in zip(gold_list, test_list):
        pcounts += _precision(goldalign=goldalign, testalign=testalign)
        # recall is precision with no insertion/deletion and swap args
        if delete:
            test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
            gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
            rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)
        else:
            rcounts += _precision(goldalign=testalign, testalign=goldalign)

    # assert pcounts[0] == rcounts[0], "TP for precision and recall are mismatched!"

    # Compute results
    # pcounts: tpstrict,fnstrict,tplax,fnlax
    # rcounts: tpstrict,fpstrict,tplax,fplax

    if pcounts[0] + pcounts[1] == 0:
        pstrict = value_for_div_by_0
    else:
        pstrict = pcounts[0] / float(pcounts[0] + pcounts[1])

    if pcounts[2] + pcounts[3] == 0:
        plax = value_for_div_by_0
    else:
        plax = pcounts[2] / float(pcounts[2] + pcounts[3])

    if rcounts[0] + rcounts[1] == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = rcounts[0] / float(rcounts[0] + rcounts[1])

    if rcounts[2] + rcounts[3] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = rcounts[2] / float(rcounts[2] + rcounts[3])

    if (pstrict + rstrict) == 0:
        fstrict = value_for_div_by_0
    else:
        fstrict = 2 * (pstrict * rstrict) / (pstrict + rstrict)

    if (plax + rlax) == 0:
        flax = value_for_div_by_0
    else:
        flax = 2 * (plax * rlax) / (plax + rlax)

    result = dict(recall_strict=rstrict,
                  recall_lax=rlax,
                  precision_strict=pstrict,
                  precision_lax=plax,
                  f1_strict=fstrict,
                  f1_lax=flax)

    return result

def score_separate(gold_alignment_list, test_alignment_list, value_for_div_by_0=0.0):
    max_src_align = 0
    max_trg_align = 0
    gold_list = defaultdict(lambda: [])
    for gold_alignments in gold_alignment_list:
        for alignment in gold_alignments:
            gold_list[(len(alignment[0]), len(alignment[1]))].append(alignment)
            max_src_align = max([max_src_align, len(alignment[0])])
            max_trg_align = max([max_trg_align, len(alignment[1])])

    test_list = defaultdict(lambda: [])
    for test_alignments in test_alignment_list:
        for alignment in test_alignments:
            test_list[(len(alignment[0]), len(alignment[1]))].append(alignment)

    res = np.full((max_src_align + 1, max_trg_align + 1), None)
    for num_src in range(max_src_align + 1):
        for num_trg in range(max_trg_align + 1):
            ret = score_multiple(gold_list=[gold_list[(num_src, num_trg)]],
                                 test_list=[test_list[(num_src, num_trg)]], delete=False)
            res[num_src][num_trg] = "{precision_strict:.3f}/{recall_strict:.3f}/{f1_strict:.3f} ".format(**ret) + "({})".format(len(gold_list[(num_src, num_trg)]))

    return res


def log_final_scores(res):
    print(' ---------------------------------', file=sys.stderr)
    print('|             |  Strict |    Lax  |', file=sys.stderr)
    print('| Precision   |   {precision_strict:.3f} |   {precision_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Recall      |   {recall_strict:.3f} |   {recall_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| F1          |   {f1_strict:.3f} |   {f1_lax:.3f} |'.format(**res), file=sys.stderr)
    print(' ---------------------------------', file=sys.stderr)

def log_matrix(res):
    from tabulate import tabulate

    header = ["trg/src"] + list(range(len(res) + 1))
    table = []
    for num_trg in range(len(res[0])):
        row_result = [str(num_trg)]
        for num_src in range(len(res)):
            row_result.append(res[(num_src, num_trg)])
        table.append(row_result)

    print(tabulate(table, header), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        'Compute strict/lax precision and recall for one or more pairs of gold/test alignments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--test', type=str, nargs='+', required=True,
                        help='one or more test alignment files')

    parser.add_argument('-g', '--gold', type=str, nargs='+', required=True,
                        help='one or more gold alignment files')

    args = parser.parse_args()

    if len(args.test) != len(args.gold):
        raise Exception('number of gold/test files must be the same')

    gold_list = [read_alignments(x) for x in args.gold]
    test_list = [read_alignments(x) for x in args.test]

    res = score_multiple(gold_list=gold_list, test_list=test_list)
    log_final_scores(res)
    res = score_separate(gold_list, test_list)
    log_matrix(res)


if __name__ == '__main__':
    main()
