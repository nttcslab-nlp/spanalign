# -*- coding: utf-8 -*-
'''
Created on 2012/09/26
update: 2019/01/24
  for python 3.6
'''

import os.path
import glob
import codecs
import shutil
import re

CTRL_CODE_PAT = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def getAbsolutePath(path_str):
    return os.path.abspath(os.path.expanduser(path_str))
# end

def getBaseName(path_str):
    return os.path.basename(getAbsolutePath(path_str))
# end

def getFileNameRoot(path_str, with_ext=False):
    (root, ext) = os.path.splitext(getBaseName(path_str))

    if (with_ext):
        ext = ext[1:]
        return (root, ext)
    else:
        return root
    # end
# end

def listUpFiles(dir_name, pattern, sort=True):
    path = os.path.join(dir_name, pattern)
    fullpath = os.path.abspath(os.path.expanduser(path))

    filelist = glob.glob(fullpath)

    if (sort):
        filelist.sort()
    # end

    return filelist
# end

def saveStringToFile(string, filename, enc="utf-8", mode="ow", add_terminator=False):
    abs_filename = getAbsolutePath(filename)

    if (os.path.exists(abs_filename)):
        if (mode == "bk"):
            (path, fname) = os.path.split(abs_filename)

            backups = listUpFiles(path, "{0}.old_*".format(fname))

            if (len(backups) == 0):
                next_num = 0
            else:
                try:
                    current_num = int(backups[-1].replace("{0}.old_".format(abs_filename), ""))
                    next_num = current_num + 1
                except ValueError:
                    next_num = 0
                # end
            # end

            if (next_num > 99):
                next_num = 0
            # end

            backup_name = "{0}.old_{1:02d}".format(abs_filename, next_num)
            shutil.copy2(abs_filename, backup_name)
        elif (mode == "st"):
            print("{0} is already exist.".format(abs_filename))
            return
        # end
    # end

    if (add_terminator and string[-1] != "\n"):
        string = "{}\n".format(string)
    # end

    with codecs.open(abs_filename, mode="w", encoding=enc) as fileObj:
        fileObj.write(string)
        fileObj.flush()
    # end
# end

def saveListToFile(stringList, filename, enc="utf-8", mode="ow"):
    if (len(stringList) == 0 or stringList[-1] != ""):
        stringList.append("")
    # end
    txt = "\n".join(stringList)

    saveStringToFile(txt, filename, enc, mode)
# end


def loadFileToList(filename, enc="utf-8-sig", commentMark="#", removeEmptyLine=True, strip="lr", line_num=0, removeCtrlCode=False):
    line_counter = 0
    results = []
    fullpath = getAbsolutePath(filename)
    with codecs.open(fullpath, "r", enc, "replace") as fileObj:
        for line in fileObj:
            line = line.replace("\0", "")

            if (strip == "lr"):
                line = line.strip()
            elif (strip == "l"):
                line = line.lstrip()
            elif (strip == "r"):
                line = line.rstrip()
            # end

            if (removeCtrlCode):
                line = CTRL_CODE_PAT.sub("", line)
            # end

            if (removeEmptyLine and not line):
                continue
            # end

            if (commentMark and line.startswith(commentMark)):
                continue
            else:
                results.append(line)
            # end

            # 取得行数指定(1以上)があった場合、その行数だけ取得したら終了
            line_counter += 1
            if (line_num > 0 and line_counter > line_num):
                break
            # end
        # end
    # end

    return results
# end

def loadFileLineIter(filename, enc="utf-8-sig", commentMark="#", removeEmptyLine=True, strip="lr", line_num=0, removeCtrlCode=False):
    fullpath = getAbsolutePath(filename)
    with codecs.open(fullpath, "r", enc, "replace") as fileObj:
        for line in fileObj:
            line = line.replace(u"\0", u"")

            if (strip == "lr"):
                line = line.strip()
            elif (strip == "l"):
                line = line.lstrip()
            elif (strip == "r"):
                line = line.rstrip()
            # end

            if (removeCtrlCode):
                line = CTRL_CODE_PAT.sub(u"", line)
            # end

            if (removeEmptyLine and not line):
                continue
            # end

            if (commentMark and line.startswith(commentMark)):
                continue
            else:
                yield line
            # end
        # end
    # end
# end

def loadFileString(filename, enc="utf-8-sig", commentMark=None, removeEmptyLine=False, strip="r", line_num=0, removeCtrlCode=False):
    lines = loadFileToList(filename, enc, commentMark, removeEmptyLine, strip, line_num, removeCtrlCode)
    lines.append("")
    result_str = "\n".join(lines)

    return result_str
# end

def collectiveReplace(files, old, new, backup=False):
    for fileName in files:
        lines = loadFileToList(fileName, commentMark=None, removeEmptyLine=False, strip="r")
        lines_str = "\n".join(lines)

        if (backup):
            bkfile = "{}.bak".format(fileName)
            saveStringToFile(lines_str, bkfile)
        # end

        lines_str = lines_str.replace(old, new)
        saveStringToFile(lines_str, fileName)
    # end
# end

def countFileLines(filename, enc="utf-8-sig"):
    fullpath = getAbsolutePath(filename)
    #count = 0

    with codecs.open(fullpath, "rb", enc, "replace") as fileObj:
        count = sum(1 for _line in fileObj)
    # end

    return count
# end
