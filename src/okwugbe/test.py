import sys,os
import logging

from cvutils import Validator
class HidePrintStatement: #This originated from with the need to hide the cvutils Validator statements  
    def __init__(self):
        super(HidePrintStatement, self).__init__()
        
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._stderr= sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr= open(os.devnull, 'w')
       
        return self._original_stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr=self._stderr


utterance='my name is Chris'
with HidePrintStatement() as hp_:
    try:
        validator = Validator('zh-HK')
        print('we are here')
        utterance_validated = validator.validate(utterance)   
        if utterance_validated is not None:
            utterance = utterance_validated
    except Exception:
        pass  
   
    print(utterance)     
print('outside print')    

sp = {
            "tatar": "tt",
            "english": "en",
            "german": "de",
            "french": "fr",
            "welsh": "cy",
            "breton": "br",
            "chuvash": "cv",
            "turkish": "tr",
            "kyrgyz": "ky",
            "irish": "ga-IE",
            "kabyle": "kab",
            "catalan": "ca",
            "taiwanese": "zh-TW",
            "slovenian": "sl",
            "italian": "it",
            "dutch": "nl",
            "hakha chin": "cnh",
            "esperanto": "eo",
            "estonian": "et",
            "persian": "fa",
            "portuguese": "pt",
            "basque": "eu",
            "spanish": "es",
            "chinese": "zh-CN",
            "mongolian": "mn",
            "sakha": "sah",
            "dhivehi": "dv",
            "kinyarwanda": "rw",
            "swedish": "sv-SE",
            "russian": "ru",
            "indonesian": "id",
            "arabic": "ar",
            "tamil": "ta",
            "interlingua": "ia",
            "latvian": "lv",
            "japanese": "ja",
            "votic": "vot",
            "abkhaz": "ab",
            "cantonese": "zh-HK",
            "romansh sursilvan": "rm-sursilv"
        }

l=[]
c=[]
for k in list(sp.keys()):    
    l.append(k)
    c.append(sp[k])

di = ['---' for i in c]
dd = '|'+'|'.join(di)+'|'
cc = '|'+'|'.join(c)+'|'
ll = '|'+'|'.join(l)+'|'
print(dd)
print(cc)
print(ll)

