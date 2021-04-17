from pipelines import qg_pipeline
from pipelines import FlashcardStyle
from transformers import pipeline
from typing import List
import nltk
import re

cloze_open_tag = "<span class='cloze'>"
cloze_close_tag = "</span>"
hl_tag = " <hl> "
hl_regex = " <hl> (.+) <hl> "

class Autocards:

    def __init__(self):
        self.generate_cards = qg_pipeline('question-generation',
                                          model='valhalla/t5-base-qg-hl',
                                          ans_model='valhalla/t5-small-qa-qg-hl')

    def create_clozes(self, text: str):
        text = self.preprocess_text(text)
        cloze_contexts = self.generate_cards(text, FlashcardStyle.Cloze)
        ret = []
        for cloze_context in cloze_contexts:
            sts = nltk.sent_tokenize(cloze_context)
            cloze = next((st for st in sts if " <hl> " in st), None)
            data = {}
            data["answer"] = re.search(hl_regex, cloze).group(1)
            data["question"] = re.sub(hl_regex, self.wrap_cloze("[...]"), cloze)
            ret.append(data)
        return ret

    @staticmethod
    def wrap_cloze(text: str):
        return cloze_open_tag + text + cloze_close_tag


    def create_qas(self, text: str):
        text = self.preprocess_text(text)
        return self.generate_cards(text, FlashcardStyle.QA)

    def reformulate_cloze(self, cloze: str):
        qg_str = "generate question: " + cloze
        answer = re.search(hl_regex, cloze).group(1)
        question = self.generate_cards._generate_questions([qg_str])
        return {
            "question": question,
            "answer": answer
        }

    @staticmethod
    def preprocess_text(text: str):
        text = (text
                .replace('\xad ', '')
                .replace('\n\n', '. ')
                .replace('\r\n\r\n', '. ')
                .replace('..', '.'))
        return re.sub(r'\[\d+\]', "", text)
