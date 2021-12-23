## Training an ASR model on a low-resource language with Okwugbe

With Okwugbe, many of the main features for a good ASR model on a low-resource language have been implememnted for You.
Some of them are:
- speech augmentation using the `time_mask` and `freq_mask` default parameters.
- gradient accumulation which can be activated with `grad_acc=True` and setting `batch_multiplier` to your choice or using the default ones.


## Training marathon for the Third Nepal Winter School in AI, 2021
The aim of this fun project is to give you practical experience with training an ASR model on a low-resource language. The reality is that training ASR models on low-resource languages is more challenging than with high-resource languages. This is because end-to-end neural networks are data hungry. However, there are some techniques to help you squeeze out the best performance from your low-resource data....and Okwugbe offers all these.

For this marathon, You will need to do the following:

#### :one: Choose any low-resource language of your choice. 
- You can use any of the low-resource languages from Common Voice. Use this table as a guide to know which languages are low-resource. 
- Alternatively, you can use your own data.

#### :two: Training with OkwuGbe.

    - Make a copy of this Colab notebook to get started:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DujjQPbMjmoh3xwJJStD5VvXN1GwZQNl?usp=sharing) 
    
    - The notebook above will guide you on training your low-resource language with OkwuGbe.   
    -Try to experiment with some of the paramenters (like `n_feats`,`grad_acc`). Use [this table of parameters](https://github.com/chrisemezue/okwugbe#parameters) as a guide.

#### :three: Logging your experience    

After training (and possibly tweaking the parameters to see if you can get better metrics), then go here  to this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1LiwbLSaNa9uwAJOb1Cag-IT9iNWt0BA0HLRlscMEPis/edit?usp=sharing) and put your details:

    - Name
    - Email
    - Link to your Colab
    - Insights: what you observed or learned during the project. It could be something about the performance of the ASR model on the test inference or some changes you noticed when you tweaked some parameters.
    - Link to your model checkpoints (if possible). 
    