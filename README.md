# sentiment_transfer_network

## Generative Adversarial Network based：

Which is based on [Style Transfer from Non-Parallel Text by Cross-Alignment](https://arxiv.org/abs/1705.09655)

### Prerequisties:
  - python 3.6 or higher
  - tensroflow 1.3 or higher
  - GPU memory 6G or higher (GeForce GTX 1060 up)
  
### usage:
1. Prepare dataset：
- Step 1 : Make a directory for your dataset. <br>
`mkdir -r data/[your_dataset_name]`
- Step 2 : Put positive and negative datasets into the directory <br>
and rename them as `pos_file.txt` and `neg_file.txt` respectively.
(Every sentences in `pos_file.txt` and `neg_file.txt` are taken apart by `\n`)
- Step 3 : Put the testing data into the directory <br>
and rename it as `seq2seq.txt`. (format of `seq2seq.txt` is the as `pos_file.txt` and `neg_file.txt`)


2. Training：
- run `python main.py -train -attention -model [your_model_name] -data_path [your_dataset_name]`
3. Testing：
- run `python main.py -test  -attention -model [your_model_name] -data_path [your_dataset_name]`
  


