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
- Step 2 : Prepare training data. <br>
Put positive and negative datasets into the directory and rename them as `pos_file.txt` and `neg_file.txt` respectively.<br>
(Every sentences in `pos_file.txt` and `neg_file.txt` are split by `\n`)
- Step 3 : Prepare testing data. <br>
Put the testing data into the directory and rename it as `seq2seq.txt`.<br>
(format of `seq2seq.txt` is the same as `pos_file.txt` and `neg_file.txt`)

2. Training：
- run `python main.py -train -attention -model [your_model_name] -data_path [your_dataset_name]`

3. Testing：
- run `python main.py -test  -attention -model [your_model_name] -data_path [your_dataset_name]`
  
## Transfer Network based：

### Prerequisties:
  - python 3.6 or higher
  - tensroflow 1.0
  
### usage:
1. Prepare dataset：
- Step 1 : Make a directory for your model. <br>
`mkdir -r works/[your_model_name]`

- Step 2 : Prepare sentiment data. <br>
Put positive and negative datasets into the directory and rename them as `pos_file.txt` and `neg_file.txt` respectively.<br>
(Every sentences in `pos_file.txt` and `neg_file.txt` are split by `\n`)

- Step 3 : Prepare dialogue data. <br>
Put dialogue dataset into the directory and rename them as `chat.txt`.<br>
(Each pair of dialogue (question and answer) is split by `\n`)

- Step 4 : Prepare testing data. <br>
Put the testing data into the root directory and rename it as `seq2seq.txt`.<br>
(format of `seq2seq.txt` is the same as `pos_file.txt` and `neg_file.txt`)

2. Training：
- Step 1： Training the variational autoencoder.
-- run `python main.py -step1 -model [your_model_name]`
- Step 2： Training the sentiment classifier.
-- run `python main.py -step2 -model [your_model_name]`
- Step 3： Training the transfer network.
-- run `python main.py -step3 -model [your_model_name]`

3. Testing：
- run `python main.py --test -model [your_model_name]`

