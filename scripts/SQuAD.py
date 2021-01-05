# -*- coding: utf-8 -*-
'''
Created on 2019/10/31

'''

class SQuAD():
    def __init__(self, version2=False):
        self.version = "v2.0" if version2 else "1.1"
        self.data = []
    # end
    
    def _dataToJson(self):
        return [d.toJson() for d in self.data]
    # end
    
    def appendData(self, dat):
        self.data.append(dat)
    # end
    
    def toJson(self):
        jo = {"version": self.version, "data": self._dataToJson()}
        
        return jo
    # end
# end

class SQuADDataItem():
    def __init__(self, title):
        self.title = title
        self.paragraphs = []
    # end
    
    def _paragraphsToJson(self):
        return [p.toJson() for p in self.paragraphs]
    # end
    
    def appendParagraph(self, p):
        self.paragraphs.append(p)
    # end
    
    def toJson(self):
        jo = {"title": self.title, "paragraphs": self._paragraphsToJson()}
        
        return jo
    # end
# end

class SQuADParagrap():
    def __init__(self, context):
        self.context = context
        self.qas = []
    # end
    
    def _qasToJson(self):
        return [qa.toJson() for qa in self.qas]
    # end
    
    def appendQA(self, qa):
        self.qas.append(qa)
    # end
    
    def extendQA(self, qa_list):
        self.qas.extend(qa_list)
    # end
    
    def toJson(self):
        jo = {"context": self.context, "qas": self._qasToJson()}
        
        return jo
    # end
# end

class SQuAD_QA():
    def __init__(self, qa_id, question, is_impossible=False):
        self.id = qa_id
        self.question = question
        self.answers = []
        self.is_impossible = is_impossible
    # end
    
    def _answersToJson(self):
        return [ans.toJson() for ans in self.answers]
    # end
    
    def appendAnswer(self, answer):
        self.answers.append(answer)
    # end
    
    def toJson(self):
        jo = {"id": self.id, 
              "question": self.question,
              "answers": self._answersToJson(),
              "is_impossible": self.is_impossible}
        
        return jo
    # end
# end
        
class SQuADAnswer():
    def __init__(self, text, start, end=None):
        self.text = text
        self.answer_start = start
        if end is None:
            self.answer_end = None
        else:
            self.answer_end = end
        # end
    # end
    
    def toJson(self):
        if self.answer_end is None:
            jo = {"text": self.text, "answer_start": self.answer_start}
        else:
            jo = {"text": self.text, "answer_start": self.answer_start, "answer_end": self.answer_end}
        # end
        
        return jo
    # end
# end

class Span():
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text
    # end
# end
