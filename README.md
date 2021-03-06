# Tree_LSTM_CS584

The folder contains:

- datasets: kaggle tweets, covid19 April sample;

- staticData: fine-tuned model (the pre-trained models are too big, can generate from the script)

- all .py files: utils, model, pre-processers

- main files notebook: rnn-lstm, tree-lstm, judgers, streaming application


Some remarks regarding running the project:

1. The Tree-LSTM is built with Deep Graph Learning(DGL), please refer to:

	https://docs.dgl.ai/en/0.4.x/tutorials/models/2_small_graph/3_tree-lstm.html;

	for set-up;

2. It is reommended all training/testing to be run with GPU, in particualr, treeLSTM taking long time to train;

3. Converting Text2Tree requires Stanford CoreNLP (download stanford-corenlp-full-2020-04-20 package), the batch processing uses:

	java -cp "*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file "tree_covid.txt" -output PENNTREES | Out-File "output.txt"  

	
4. The judgers requires bit set-up, please refer to:
	
	IBM: https://www.ibm.com/cloud/watson-natural-language-understanding
	Stanford: https://github.com/stanfordnlp/python-stanford-corenlp

5. The staticData folder has pre-trained models saved, and you need to download glove 100d, 200d to run most of scripts

6. The streaming platform is now linked to TextBlob, to deploy TreeLSTM or RNNLSTM, it needs bit setup (convert text2tree in realtime), we are not uploading all supporting materials, but just a framework.


