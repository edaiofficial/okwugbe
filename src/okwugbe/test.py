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