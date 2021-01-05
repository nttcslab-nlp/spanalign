# -*- coding: utf-8 -*-
'''
Created on 2019/12/11

人手で作成した *.pair ファイルと元テキスト（1行1文）から文対応実験用 SQuAD データを作成する。

pair_id: 対訳データのファイル名に付けられた 6 桁の番号
qa_id: pair_id + 文対応番号（対訳データ毎に 000000 からの連番）
context: 元テキストを一纏めにしたもの。改行はどうする？
    -> 無視する（改行なしで繋げる）
    -> 特別なトークンを入れる
    -> \n のまま1文字として処理する（ただしJSON上ではエスケープされて \\n になる）

SQuAD は 2.0 に対応。対応無しの文もデータに入れる。（is_impossible = True）


'''

import argparse
import json
import sys
from pathlib import Path
from ast import literal_eval

import fileUtils
import SQuAD

class SpanInfo():
    def __init__(self, head=None, span=None, pair_head=None):
        self.modify(head, span, pair_head)
    # end

    def modify(self, head, span, pair_head=None):
        self.head = head
        self.span = span
        self.pair_head = pair_head
    # end
# end

def lineNumCsvToLineIdxs(lines_csv, zero_based_index=False):
    '''
        文番号の CSV 文字列を文インデックスのリストにする
        ただし文番号が連続していない場合は空のリストを返す
    '''
    offset = 0 if zero_based_index else 1

    if len(lines_csv) == '[]':
        return []

    line_idx_list = literal_eval(lines_csv)
    s_l_idx_list = [int(x)-offset for x in line_idx_list] # offset 分ずらして 0 始まりにする
    s_l_idx_list.sort()

    # 連続しているなら最大インデックスは先頭インデックス＋リストの長さ−１になるはず
    idx_max = s_l_idx_list[-1]
    if idx_max != s_l_idx_list[0] + len(s_l_idx_list) - 1:
        return []
    else:
        return s_l_idx_list
    # end
# end

def makeRawLineToCharSpanMap(raw_lines, line_sep=""):
    '''
        元テキストの各文（インデックス）を文字 Span に対応づける
        文区切り記号については line_sep で指定できるようにし、その分も考慮して Span の開始位置を算出
        元テキストの先頭を 0 文字目とする
    '''
    raw_line_to_span = []
    sep_len = len(line_sep)

    offset = 0
    for line in raw_lines:
        l_len = len(line)
        span = SQuAD.Span(offset, offset+l_len-1, line)
        raw_line_to_span.append(span)

        offset = offset + l_len + sep_len
    # end

    return raw_line_to_span
# end

def lineIndexesToSpan(sent_line_indexes, raw_line_idx_to_span_map, line_sep=""):
    '''
       sent_line_indexes: 1対応を構成する片側の文インデックスのリスト
       raw_line_idx_to_span_map: 文インデックスと Span の対応マップ
       line_sep: 文区切り記号
    '''
    start = raw_line_idx_to_span_map[sent_line_indexes[0]].start
    end = raw_line_idx_to_span_map[sent_line_indexes[-1]].end

    span_texts = []
    for line_idx in sent_line_indexes:
        span_texts.append(raw_line_idx_to_span_map[line_idx].text)
    # end
    text = line_sep.join(span_texts)

    return SQuAD.Span(start, end, text)
# end

def setLineToSpan(line_to_sent_span, line_idxs, raw_line_to_span, line_sep):
    '''
        行：Span 対応リストの span 開始行に対応する要素の SpanInfo オブジェクトを変更し、span 開始行のインデックスを返す
    '''
    head_idx = line_idxs[0]
    char_span = lineIndexesToSpan(line_idxs, raw_line_to_span, line_sep)
    line_to_sent_span[head_idx].modify(head=head_idx, span=char_span)

    for idx in line_idxs[1:]:
        line_to_sent_span[idx].modify(head=-1, span=None, pair_head=-1)
    # end

    return head_idx
# end

def makeSentSpanList(pair_line, raw_lines_l1, raw_lines_l2, line_sep="", zero_based_index=False):
    '''
        文対応情報と元テキストから文字単位 Span と対応相手の情報を持った辞書データを作成し、
        元テキストの文インデックスと対応付けて格納したリストを返す
        1-n の対応の時、n が連続した文でないなら、その対応は無視する
    '''
    line_to_sent_span_l1 = [SpanInfo(head=None, span=SQuAD.Span(-1, -1, line)) for line in raw_lines_l1]
    line_to_sent_span_l2 = [SpanInfo(head=None, span=SQuAD.Span(-1, -1, line)) for line in raw_lines_l2]

    raw_line_to_span_l1 = makeRawLineToCharSpanMap(raw_lines_l1, line_sep)
    raw_line_to_span_l2 = makeRawLineToCharSpanMap(raw_lines_l2, line_sep)

    for line in pair_line:
        lines_csv_l1, lines_csv_l2 = line.split(":")

        # 文対 CSV を文インデックスのリストにする
        line_idxs_l1 = lineNumCsvToLineIdxs(lines_csv_l1, zero_based_index)
        line_idxs_l2 = lineNumCsvToLineIdxs(lines_csv_l2, zero_based_index)

        # line_idxs が両方空リストでないなら
        if line_idxs_l1 and line_idxs_l2:
            span_head_idx_l1 = setLineToSpan(line_to_sent_span_l1, line_idxs_l1, raw_line_to_span_l1, line_sep)
            span_head_idx_l2 = setLineToSpan(line_to_sent_span_l2, line_idxs_l2, raw_line_to_span_l2, line_sep)

            # 対応する相手側の開始行インデックスを保持
            line_to_sent_span_l1[span_head_idx_l1].pair_head = span_head_idx_l2
            line_to_sent_span_l2[span_head_idx_l2].pair_head = span_head_idx_l1
        # end
    # end

    return line_to_sent_span_l1, line_to_sent_span_l2
# end

def makeContextText(raw_lines, line_sep=""):
    '''
        元データを文区切り記号で繋げて context 用テキストを作る
    '''
    return line_sep.join(raw_lines)
# end

def loadSpansAndContext(pairs_file, raw_file_l1, raw_file_l2, line_sep="", zero_based_index=False):
    '''
        データをファイルから読み出し、元テキスト行：span の対応リストと context 用テキストを返す
    '''
    pair_lines = fileUtils.loadFileToList(pairs_file)
    raw_lines_l1 = fileUtils.loadFileToList(raw_file_l1, commentMark=None)
    raw_lines_l2 = fileUtils.loadFileToList(raw_file_l2, commentMark=None)

    # 全角スペースを半角に
    raw_lines_l1 = [x.replace("　", " ") for x in raw_lines_l1]
    raw_lines_l2 = [x.replace("　", " ") for x in raw_lines_l2]

    line_to_sent_span_l1, line_to_sent_span_l2 = makeSentSpanList(pair_lines, raw_lines_l1, raw_lines_l2, line_sep, zero_based_index)

    context_l1 = makeContextText(raw_lines_l1, line_sep)
    context_l2 = makeContextText(raw_lines_l2, line_sep)

    return line_to_sent_span_l1, line_to_sent_span_l2, context_l1, context_l2
# end

def makeQA(line_to_sent_span_Q, line_to_sent_span_A, qa_id_prefix):
    qa_list = []

    for line_idx, q_span_info in enumerate(line_to_sent_span_Q):
        if q_span_info.head == -1:
            continue
        elif q_span_info.head is None:
            qa_id = "{}_{}_X".format(qa_id_prefix, line_idx)
            q_str = q_span_info.span.text
            qa = SQuAD.SQuAD_QA(qa_id, q_str, is_impossible=True)
        else:
            qa_id = "{}_{}".format(qa_id_prefix, line_idx)
            q_str = q_span_info.span.text
            qa = SQuAD.SQuAD_QA(qa_id, q_str, is_impossible=False)

            ans_head = q_span_info.pair_head
            ans_span = line_to_sent_span_A[ans_head].span
            answer = SQuAD.SQuADAnswer(ans_span.text, ans_span.start, ans_span.end)

            qa.appendAnswer(answer)
        # end

        qa_list.append(qa)
    # end

    return  qa_list
# end

def makeParagraph(line_to_sent_span_l1, line_to_sent_span_l2, context_l1, context_l2, pid_prefix):
    paragraph_l1_to_l2 = SQuAD.SQuADParagrap(context_l2)
    paragraph_l2_to_l1 = SQuAD.SQuADParagrap(context_l1)

    qa_id_prefix_l1tol2 = "{}_1_2".format(pid_prefix)
    qa_list_l1tol2 = makeQA(line_to_sent_span_l1, line_to_sent_span_l2, qa_id_prefix_l1tol2)
    qa_id_prefix_l2tol1 = "{}_2_1".format(pid_prefix)
    qa_list_l2tol1 = makeQA(line_to_sent_span_l2, line_to_sent_span_l1, qa_id_prefix_l2tol1)

    paragraph_l1_to_l2.extendQA(qa_list_l1tol2)
    paragraph_l2_to_l1.extendQA(qa_list_l2tol1)

    return paragraph_l1_to_l2, paragraph_l2_to_l1
# end


def main(args):
    pairs_file = Path(args.pairs_file)
    raw_file_l1 = args.raw_file_l1
    raw_file_l2 = args.raw_file_l2
    dest = args.dest
    title_prefix = args.title_prefix
    line_sep = args.line_sep
    indent = args.indent
    zero_based_index = args.zero_based_index

    title = "{}_{}".format(title_prefix, pairs_file.stem)

    line_to_sent_span_l1, line_to_sent_span_l2, context_l1, context_l2 = loadSpansAndContext(pairs_file, raw_file_l1, raw_file_l2, line_sep, zero_based_index)
    paragraph_l1_to_l2, paragraph_l2_to_l1 = makeParagraph(line_to_sent_span_l1, line_to_sent_span_l2, context_l1, context_l2, title)

    squad_data = SQuAD.SQuADDataItem(title)
    squad_data.appendParagraph(paragraph_l1_to_l2)
    squad_data.appendParagraph(paragraph_l2_to_l1)

    squad = SQuAD.SQuAD(version2=True)
    squad.appendData(squad_data)

    with open(dest, "w", encoding='utf-8') as fp:
        json.dump(squad.toJson(), fp, ensure_ascii=False, indent=indent)
    # end
# end

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs_file", help="input *.pair file")
    parser.add_argument("raw_file_l1", help="source raw text file")
    parser.add_argument("raw_file_l2", help="target raw text file")
    parser.add_argument("dest", help="output file")
    parser.add_argument("title_prefix", help="prefix for SQuAD title")
    parser.add_argument("-s", dest="line_sep", default="\n", help="line separator")
    parser.add_argument("-i", dest="indent", type=int, default=2, help="indent level for JSON dump")
    parser.add_argument("-z", dest="zero_based_index", action="store_true", help="whether sentence numbers in *.pair file are zero-based")

    args = parser.parse_args()
    main(args)
# end
