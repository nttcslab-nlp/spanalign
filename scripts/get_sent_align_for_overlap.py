# -*- coding: utf-8 -*-

"""
created by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
updated on Dec. 22, 2020 by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
"""

import sys
import os
import re
import json
import subprocess
import tempfile

from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from statistics import mean
from typing import List

CPLEX_PATH = "__SET_CPLEX_PATH__"


def edit_distance(s1, s2):
    len1 = len(s1)
    len2 = len(s2)

    lev = [[0] * len2 for _ in range(len1)]
    for i in range(len1):
        lev[i][0] = i
    for j in range(len2):
        lev[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            lev[i][j] = min(
                lev[i - 1][j] + 1,
                lev[i][j - 1] + 1,
                lev[i - 1][j - 1] + (s1[i - 1] != s2[j - 1])
            )
    return lev[len1][len2]


def load_raw_data(filepath: Path) -> List[str]:
    sents = []
    with filepath.open() as ifs:
        for line in ifs:
            sents.append(line.strip())

    return sents

class Span(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __len__(self):
        if self.start < 0:
            return 0
        return self.end - self.start + 1


Alignment = namedtuple("Alignment", ['l1', 'l2', 'prob'])
def get_alignment_candidate(trg_sents: List[str],
                            nbest_predictions: dict,
                            num_src_sents: int,
                            qa_id_prefix: str,
                            nbest: int=1,
                            max_sents: int=-1,
                            reverse: bool=False) -> List[Alignment]:
    noans_flag = [False] * num_src_sents
    candidates = []
    for pos in range(len(trg_sents)):
        trg_span = Span(pos, pos)
        src_span = Span(-1, -1)
        if reverse:
            candidates.append(Alignment(trg_span, src_span, 0))
        else:
            candidates.append(Alignment(src_span, trg_span, 0))

    for k, v in nbest_predictions.items():
        if not k.startswith(qa_id_prefix):
            continue

        start, end = map(int, k.split("_")[-1].split("-"))
        src_span = Span(start, end)

        for i in range(min(len(v), nbest)):
            pred = v[i]
            pred_text = pred['text']
            pred_prob = pred['probability']

            def span_dist(span):
                s2 = "\n".join(trg_sents[span.start : span.end + 1])
                return edit_distance(pred_text, s2)

            trg_span = Span(-1, -1)
            if len(pred_text) != 0:
                start = 0
                trg_span = Span(-1, -1)
                while start < len(trg_sents):
                    if pred_text.find(trg_sents[start]) < 0:
                        start += 1
                        continue

                    end = start + 1
                    while (end < len(trg_sents) and
                           pred_text.find(trg_sents[end]) >= 0):
                        end += 1

                    cand = Span(start, end - 1)
                    if trg_span.start < 0 or span_dist(trg_span) > span_dist(cand):
                        trg_span = cand
                    start = end + 1

            if len(src_span) > 1 and trg_span.start == -1:
                continue
            if max_sents > 0 and len(trg_span) > max_sents:
                continue
            if len(src_span) == 1 and trg_span.start == -1:
                noans_flag[src_span.start] = True

            if reverse:
                candidates.append(Alignment(trg_span, src_span, pred_prob))
            else:
                candidates.append(Alignment(src_span, trg_span, pred_prob))

    for pos, flag in enumerate(noans_flag):
        if flag:
            continue

        src_span = Span(pos, pos)
        trg_span = Span(-1, -1)
        if reverse:
            candidates.append(Alignment(trg_span, src_span, 0))
        else:
            candidates.append(Alignment(src_span, trg_span, 0))

    return candidates

def fix_variable_name(src, trg):
    name = "x%d_%d_%d_%d" % (src.start, src.end, trg.start, trg.end)
    name = name.replace("-1", "X")

    return name

def create_lp(alignment_candidates, output_file, sentence_penalty=0.):
    ofs = output_file.open('w')
    src_pos = set()
    trg_pos = set()

    print("Minimize", file=ofs)
    print('obj:', file=ofs)
    for alignment in alignment_candidates:
        src = alignment.l1
        trg = alignment.l2
        prob = alignment.prob

        score = ((1 - prob) + sentence_penalty) * ((max(1, len(src)) + max(1, len(trg))) / 2)
        if score >= 0:
            print("+", file=ofs, end='')
        print(score, fix_variable_name(src, trg), file=ofs)
        src_pos.add(src.start)
        src_pos.add(src.end)
        trg_pos.add(trg.start)
        trg_pos.add(trg.end)
    print('', file=ofs)

    print('Subject to', file=ofs)
    print('', file=ofs)
    for pos in src_pos:
        if pos < 0:
            continue

        print('src_%d:' % (pos), file=ofs)
        for alignment in alignment_candidates:
            src = alignment.l1
            trg = alignment.l2
            if src.start <= pos <= src.end:
                print("+ 1", fix_variable_name(src, trg), file=ofs)
        print("= 1", file=ofs)
        print("", file=ofs)

    for pos in trg_pos:
        if pos < 0:
            continue

        print("trg_%d:" % (pos), file=ofs)
        for alignment in alignment_candidates:
            src = alignment.l1
            trg = alignment.l2
            if trg.start <= pos <= trg.end:
                print("+ 1", fix_variable_name(src, trg), file=ofs)
        print('= 1', file=ofs)
        print("", file=ofs)

    print("Binary", file=ofs)
    for alignment in alignment_candidates:
        src = alignment.l1
        trg = alignment.l2
        print(fix_variable_name(src, trg), file=ofs)

    print("End", file=ofs)
    ofs.close()

def solve_alignment(alignment_candidates: List[Alignment], title: str,
                    sentence_penalty: float=0, use_doc_score: bool=False):
    lp_file = Path('/tmp/solve_{}.lp'.format(title))
    create_lp(alignment_candidates, lp_file, sentence_penalty)

    batch_file = Path('/tmp/solve_{}.batch'.format(title))
    with batch_file.open('w') as ofs:
        print("read", str(lp_file), file=ofs)
        print("optimize", file=ofs)
        print("display solution variables x*", file=ofs)

    command = [CPLEX_PATH, '-f', batch_file]
    result = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8').stdout

    solution_span = []
    doc_score = 0
    for line in result.split('\n'):
        if len(line) == 0 or line[0] != 'x':
            continue

        a, b, c, d, _ = re.split('[ _]+', line[1:])
        a, b, c, d = [int(x) if x != "X" else -1 for x in [a, b, c, d]]

        sent_prob = 0
        for cand in alignment_candidates:
            if cand.l1.start != a or cand.l1.end != b or cand.l2.start != c or cand.l2.end != d:
                continue
            sent_prob = max(sent_prob, cand.prob)
        score = (1 - sent_prob + sentence_penalty) * ((max(1, len(Span(a, b))) + max(1, len(Span(c, d)))) / 2)

        doc_score += score
        solution_span.append((Span(a, b), Span(c, d), score))

    doc_score /= len(solution_span)
    for idx in range(len(solution_span)):
        l1, l2, score = solution_span[idx]
        if use_doc_score:
            score *= doc_score
        solution_span[idx] = (l1, l2, score)

    assert len(solution_span) > 0, 'solution span not found.'
    lp_file.unlink()
    batch_file.unlink()

    return solution_span

def output_alignments(alignments, e_lines, f_lines, index_offset=1,
                      sentence_penalty=0, ofs=sys.stdout):
    for src, trg, score in alignments:
        src_sid = list(map(str, range(src.start + index_offset, src.end + 1 + index_offset)))
        trg_sid = list(map(str, range(trg.start + index_offset, trg.end + 1 + index_offset)))

        if src.start > -1 and trg.start > -1:
            print("[{}]:[{}]:{:.04f}".format(",".join(src_sid), ",".join(trg_sid), score),
                  file=ofs)
        elif src.start > -1:
            for sid in src_sid:
                print("[{}]:[]:{:.04f}".format(sid, score),
                      file=ofs)
        else:
            for sid in trg_sid:
                print("[]:[{}]:{:.04f}".format(sid, score),
                      file=ofs)

def merge_candidate(alignments_1, alignments_2):
    align_set = defaultdict(lambda: [0., 0.])

    for align in alignments_1:
        align_set[(align.l1.start, align.l1.end, align.l2.start, align.l2.end)][0] = align.prob

    for align in alignments_2:
        align_set[(align.l1.start, align.l1.end, align.l2.start, align.l2.end)][1] = align.prob

    alignments = []
    for k, v in align_set.items():
        src = Span(k[0], k[1])
        trg = Span(k[2], k[3])
        v = mean(v)
        alignments.append(Alignment(src, trg, v))

    return alignments

def main(config):
    offset = 0 if config.zero_index else 1

    with config.nbest_predictions.open() as ifs:
        nbest_predictions = json.load(ifs)

    sents_l1 = load_raw_data(config.lang1)
    sents_l2 = load_raw_data(config.lang2)

    l1_to_l2_prefix = config.title + "_1_2"
    l1_to_l2_candidate = get_alignment_candidate(sents_l2, nbest_predictions,
                                                 len(sents_l1), l1_to_l2_prefix,
                                                 nbest=config.nbest,
                                                 max_sents=config.max_sents)
    l1_to_l2_alignments = solve_alignment(l1_to_l2_candidate, config.title,
                                          config.sentence_penalty, config.use_doc_score)
    with config.output.with_suffix('.e2f.pair').open('w') as ofs:
        output_alignments(l1_to_l2_alignments, len(sents_l1), len(sents_l2), offset,
                          ofs=ofs)

    l2_to_l1_prefix = config.title + "_2_1"
    l2_to_l1_candidate = get_alignment_candidate(sents_l1, nbest_predictions,
                                                 len(sents_l2), l2_to_l1_prefix,
                                                 nbest=config.nbest,
                                                 max_sents=config.max_sents,
                                                 reverse=True)
    l2_to_l1_alignments = solve_alignment(l2_to_l1_candidate, config.title,
                                          config.sentence_penalty, config.use_doc_score)
    with config.output.with_suffix('.f2e.pair').open('w') as ofs:
        output_alignments(l2_to_l1_alignments, len(sents_l1), len(sents_l2), offset,
                          ofs=ofs)

    bidi_candidate = merge_candidate(l1_to_l2_candidate, l2_to_l1_candidate)
    bidi_alignments = solve_alignment(bidi_candidate, config.title,
                                      config.sentence_penalty, config.use_doc_score)
    with config.output.with_suffix('.bidi.pair').open('w') as ofs:
        output_alignments(bidi_alignments, len(sents_l1), len(sents_l2), offset,
                          ofs=ofs)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('lang1', type=Path)
    parser.add_argument('lang2', type=Path)
    parser.add_argument('nbest_predictions', type=Path)
    parser.add_argument('title', type=str)
    parser.add_argument('output', type=Path)
    parser.add_argument('-z', '--zero_index', action='store_true',
                        help='output alignment based on 0-index')
    parser.add_argument('-n', '--nbest', type=int, default=1)
    parser.add_argument('-m', '--max_sents', type=int, default=-1)
    parser.add_argument('-p', '--sentence_penalty', type=float, default=0.)
    parser.add_argument('-d', '--use_doc_score', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    config = parse_args()

    print(config.title)
    main(config)
