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
---
size_categories:
  ab:
  - n<1K
  ar:
  - 10K<n<100K
  as:
  - n<1K
  br:
  - 10K<n<100K
  ca:
  - 100K<n<1M
  cnh:
  - 1K<n<10K
  cs:
  - 10K<n<100K
  cv:
  - 10K<n<100K
  cy:
  - 10K<n<100K
  de:
  - 100K<n<1M
  dv:
  - 1K<n<10K
  el:
  - 10K<n<100K
  en:
  - 100K<n<1M
  eo:
  - 10K<n<100K
  es:
  - 100K<n<1M
  et:
  - 10K<n<100K
  eu:
  - 10K<n<100K
  fa:
  - 10K<n<100K
  fi:
  - 1K<n<10K
  fr:
  - 100K<n<1M
  fy-NL:
  - 10K<n<100K
  ga-IE:
  - 1K<n<10K
  hi:
  - n<1K
  hsb:
  - 1K<n<10K
  hu:
  - 1K<n<10K
  ia:
  - 1K<n<10K
  id:
  - 10K<n<100K
  it:
  - 100K<n<1M
  ja:
  - 1K<n<10K
  ka:
  - 1K<n<10K
  kab:
  - 100K<n<1M
  ky:
  - 10K<n<100K
  lg:
  - 1K<n<10K
  lt:
  - 1K<n<10K
  lv:
  - 1K<n<10K
  mn:
  - 1K<n<10K
  mt:
  - 10K<n<100K
  nl:
  - 10K<n<100K
  or:
  - 1K<n<10K
  pa-IN:
  - 1K<n<10K
  pl:
  - 10K<n<100K
  pt:
  - 10K<n<100K
  rm-sursilv:
  - 1K<n<10K
  rm-vallader:
  - 1K<n<10K
  ro:
  - 1K<n<10K
  ru:
  - 10K<n<100K
  rw:
  - 100K<n<1M
  sah:
  - 1K<n<10K
  sl:
  - 1K<n<10K
  sv-SE:
  - 1K<n<10K
  ta:
  - 10K<n<100K
  th:
  - 10K<n<100K
  tr:
  - 1K<n<10K
  tt:
  - 10K<n<100K
  uk:
  - 10K<n<100K
  vi:
  - 1K<n<10K
  vot:
  - n<1K
  zh-CN:
  - 10K<n<100K
  zh-HK:
  - 10K<n<100K
  zh-TW:
  - 10K<n<100K
---
- Alternatively, you can use your own data.

#### :two: Training with OkwuGbe.

- Make a copy of this Colab notebook to get started:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DujjQPbMjmoh3xwJJStD5VvXN1GwZQNl?usp=sharing) 

- The notebook above will guide you on training your low-resource language with OkwuGbe.   
- Try to experiment with some of the paramenters (like `n_feats`,`grad_acc`). Use [this table of parameters](https://github.com/chrisemezue/okwugbe#parameters) as a guide.

#### :three: Documenting your experience    

After training (and possibly tweaking the parameters to see if you can get better metrics), then go here  to this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1LiwbLSaNa9uwAJOb1Cag-IT9iNWt0BA0HLRlscMEPis/edit?usp=sharing) and put your details:

- Name
- Email
- Link to your Colab
- Insights: what you observed or learned during the project. It could be something about the performance of the ASR model on the test inference or some changes you noticed when you tweaked some parameters.
- Link to your model checkpoints (if possible). 

##### :four: Gifts
Exciting gifts await the participants of the marathon!
We will contact you via your email to send you the gifts in a while.

______
If you liked using the Okwugbe platform, please give it a star on Github.

If there are some bugs you encountered or features you think will benefit ASR for low-resource languages, please do reach out to us. You can create a Github issue.
