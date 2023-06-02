# Supervised Contrastive Learning for Pre-Training Bioacoustic Few-Shot Systems
Autors : Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---

This is the implementation of our work for the challenge DCASE 2023 task 5

Our approach consists in :
<ul>
<li>Training a feature extractor on the training set</li>
<li>Training a linear classifier on each audio of the validation set</li>
</ul>

Firstly, we create the spectrograms of the training set :\
```create_train.py``` : with argument ```--traindir``` for the folder containing the training datasets.

To train the feature extractor :\
```train.py``` : with arguments ```--traindir``` (the same as above), ```--device``` the device to train on, and others concerning training and data augmentation hyperparameters that can be found in ```args.py``` with default values that we used.\
Train checkpoint used for the submission can be download from this link : https://drive.google.com/file/d/1ANO4oEhZ_EJvNBHz4rCE939MX08codBm/view?usp=sharing

To validate the learned feature extractor using 5-shots :\
```evaluate.py``` : with arguments ```--valdir``` for the folder containing the validation datasets, and others concerning hyperparameters that can also be found in ```args.py```. Add :\
For submission 1 :\
```--ft 0 --ftlr 0.01 --ftepochs 20 --method ce --adam```\
For submission 2 :\
```--ft 1 --ftlr 0.001 --ftepochs 40 --method ce --adam```\
For submission 3 :\
```--ft 2 --ftlr 0.001 --ftepochs 40 --method ce --adam```\
For submission 4 :\
```--ft 3 --ftlr 0.001 --ftepochs 40 --method ce --adam```

To get the scores :\
```evaluation_metrics/evaluate.py``` : with arguments ```-pred_file``` for the predictions csv file created by ```evaluate.py```, ```-ref_files``` for the path of validation datasets, and ```-save_path``` for the folder to save the scores json file

To cite this work :\
```
@techreport{Moummad2023,
    Author = "Moummad, Ilyass and Serizel, Romain and Farrugia, Nicolas",
    title = "SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINING BIOACOUSTIC FEW SHOT SYSTEMS",
    institution = "DCASE2023 Challenge",
    year = "2023",
    month = "June",
    abstract = "We show in this work that learning a rich feature extractor from scratch using only official training data is feasible. We achieve this by learning representations using a supervised contrastive learning framework. We then transfer the learned feature extractor to the sets of validation and test for few-shot evaluation. For fewshot validation, we simply train a linear classifier on the negative and positive shots and obtain a F-score of 63.46\% outperforming the baseline by a large margin. We don’t use any external data or pretrained model. Our approach doesn’t require choosing a threshold for prediction or any post-processing technique"
}
```
