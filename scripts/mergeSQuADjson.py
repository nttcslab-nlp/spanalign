# -*- coding: utf-8 -*-
'''
Created on 2019/01/25

複数の SQuDA 形式データを一つにまとめる
各 SQuAD の JSON から data 要素のリストの中身を取り出し、
出力用 JSON の data リストへ追加

'''

import argparse
import json
from pathlib import Path

import fileUtils

def makeQARefMap(squad_obj):
    qa_map = dict()

    for data in squad_obj["data"]:
        for p in data["paragraphs"]:
            for qa in p["qas"]:
                qa_id = qa["id"]

                qa_map[qa_id] = qa
            # end
        # end
    # end

    return qa_map
# end



def merge(jsonFileList, version="1.1"):
    merged = {"version": version, "data": []}
    for json_file in jsonFileList:
        with open(json_file) as f:
            jo = json.load(f)

            merged["data"].extend(jo["data"])
        # end
    # end

    return merged
# end

def main(args):
    if args.src_list:
        src_list = fileUtils.loadFileToList(args.src_list)
    elif args.src_files:
        src_list = args.src_files
    elif args.src_dir:
        if (args.mode):
            file_pat = "{}.{}.json".format(args.pat, args.mode)
        else:
            file_pat = "{}.json".format(args.pat)
        # end
        src_list = fileUtils.listUpFiles(args.src_dir, file_pat)
    else:
        print("No SRC")
        return
    # end

    print("merge files: {}".format(src_list))
    merged = merge(src_list, args.version)

    dst_file = args.dst_file if args.dst_file else args.dst_file_op
    if dst_file:
        if isinstance(dst_file, list):
            dst_file = dst_file[0]
        # end
        dst_file = Path(dst_file)
        with open(dst_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        # end
    else:
        print(json.dumps(merged))
    # end
# end

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dst_file", nargs="?", default=None)
    parser.add_argument("-c", "--src_dir", default=None)
    parser.add_argument("-s", "--src_files", dest="src_files", nargs="+")
    parser.add_argument("-d", "--dest_file", dest="dst_file_op", nargs=1)
    parser.add_argument("-l", dest="src_list", default=None)
    parser.add_argument("-r", dest="impossible_item_rate", default=None)
    parser.add_argument("--mode", dest="mode", default=None)
    parser.add_argument("--pat", dest="pat", default="*")
    parser.add_argument("--squad_version", dest="version", default="1.1")
    args = parser.parse_args()

    main(args)
# end
