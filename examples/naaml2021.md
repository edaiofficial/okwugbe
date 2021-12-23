## :runner: Training an ASR model on a low-resource language with Okwugbe
The aim of this fun project is to give you practical experience with training an ASR model on a low-resource language. The reality is that training ASR models on low-resource languages is more challenging than with high-resource languages. This is because end-to-end neural networks are data hungry. However, there are some techniques to help you squeeze out the best performance from your low-resource data....and Okwugbe offers all these. This project will enable you figure them out practically.

For this marathon, You will need to do the following:

#### :one: Choose any low-resource language of your choice. 
- You can use any of the low-resource languages from Common Voice. The table below gives the approximate size of speech-text data in each of the Common Voice langugaes. Choose any low-resource language of your choice (those with `n<1K` or `1K<n<10K`).

| ab | ar | as | br | ca | cnh | cs | cv | cy | de | dv | el | en | eo | es | et | eu | fa | fi | fr | fyNL | gaIE | hi | hsb | hu | ia | id | it | ja | ka | kab | ky | lg | lt | lv | mn | mt | nl | or | paIN | pl | pt | rmsursilv | rmvallader | ro | ru | rw | sah | sl | svSE | ta | th | tr | tt | uk | vi | vot | zhCN | zhHK | zhTW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `n<1K` | `10K<n<100K` | `n<1K` | `10K<n<100K` | `100K<n<1M` | `1K<n<10K` | `10K<n<100K` | `10K<n<100K` | `10K<n<100K` | `100K<n<1M` | `1K<n<10K` | `10K<n<100K` | `100K<n<1M` | `10K<n<100K` | `100K<n<1M` | `10K<n<100K` | `10K<n<100K` | `10K<n<100K` | `1K<n<10K` | `100K<n<1M` | `10K<n<100K` | `1K<n<10K` | `n<1K` | `1K<n<10K` | `1K<n<10K` | `1K<n<10K` | `10K<n<100K` | `100K<n<1M` | `1K<n<10K` | `1K<n<10K` | `100K<n<1M` | `10K<n<100K` | `1K<n<10K` | `1K<n<10K` | `1K<n<10K` | `1K<n<10K` | `10K<n<100K` | `10K<n<100K` | `1K<n<10K` | `1K<n<10K` | `10K<n<100K` | `10K<n<100K` | `1K<n<10K` | `1K<n<10K` | `1K<n<10K` | `10K<n<100K` | `100K<n<1M` | `1K<n<10K` | `1K<n<10K` | `1K<n<10K` | `10K<n<100K` | `10K<n<100K` | `1K<n<10K` | `10K<n<100K` | `10K<n<100K` | `1K<n<10K` | `n<1K` | `10K<n<100K` | `10K<n<100K` | `10K<n<100K` |

- Alternatively, you can use your own data.

#### :two: Training with OkwuGbe.

- Make a copy of this Colab notebook to get started:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12XiQCuQzOr7lye2sFCvsn4Ch_DNevx4u?usp=sharing) 

- The notebook above will guide you on training your low-resource language (from Common Voice) with OkwuGbe.  
- Refer to the [Usage](https://github.com/edaiofficial/okwugbe#usage) section for the general guide.  
- Try to experiment with some of the paramenters (like `n_feats`,`grad_acc`). Use [this table of parameters](https://github.com/edaiofficial/okwugbe#parameters) as a guide.

#### :three: Documenting your experience    

After training (and possibly tweaking the parameters to see if you can get better metrics), then go here  to this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1LiwbLSaNa9uwAJOb1Cag-IT9iNWt0BA0HLRlscMEPis/edit?usp=sharing) and put your details:

- Name
- Email
- Link to your Colab
- Insights: what you observed or learned during the project. It could be something about the performance of the ASR model on the test inference or some changes you noticed when you tweaked some parameters.
- Link to your model checkpoints (if possible). 

##### :four: What You Gain

The most exciting gift though is practical experience with training an ASR model on a low-resource language :smile:! 

##### :five: Contact for enquiries

In case of enquiries or issues, please contact chris.emezue@gmail.com or femipancrace.dossou@gmail.com  and we'll be sure to answer you in a jiffy!
______
If you liked using the Okwugbe platform, please give it a star on Github.

If there are some bugs you encountered or features you think will benefit ASR for low-resource languages, please do reach out to us. You can create a Github issue.
