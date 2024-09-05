# Supervised Contrastive Learning for Pre-Training Bioacoustic Few-Shot Systems
Authors: Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---

This is the implementation of our [submission](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Moummad_IMT_t5.pdf) for the challenge DCASE 2023 task 5.\
We invite you to take a look at the [workshop paper version](https://dcase.community/documents/workshop2023/proceedings/DCASE2023Workshop_Moummad_63.pdf) for more details and ablation studies.\
We ranked 2nd in the challenge. For more informations about the challenge results, [click here](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection-results) 

Our approach consists in :
<ul>
<li>Training a feature extractor on the training set</li>
<li>Training a linear classifier on each audio of the validation set</li>
</ul>

Dataset to be downloaded from this link: [DCASE 2023 TASK5 Dataset](https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection)

Firstly, we create the spectrograms of the training set :\
```create_train.py``` : with argument ```--traindir``` for the folder containing the training datasets.

To train the feature extractor :\
```train.py``` : with arguments ```--traindir``` (the same as above), ```--device``` the device to train on, and others concerning training and data augmentation hyperparameters that can be found in ```args.py``` with default values that we used.

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
```evaluation_metrics/evaluation.py``` : with arguments ```-pred_file``` for the predictions csv file created by ```evaluate.py``` (the file is in : traindir/../../outputs/eval.csv'), ```-ref_files``` for the path of validation datasets, and ```-save_path``` for the folder where to save the scores json file

If you any question or a problem with the code/results do not hesitate to mail me on : ilyass.moummad@imt-atlantique.fr or open an issue on this repository, I am very responsive.

---

### Credits
We are thankful for the challenge baseline code that helped us make this repository : https://github.com/ilyassmoummad/dcase-few-shot-bioacoustic/tree/main

### Bonus
Take a look at our newer work accepted at ICASSP 2024 where we improve the pre-training loss as well as the inference strategy (which is more stable) : https://github.com/ilyassmoummad/RCL_FS_BSED

---

### To cite the challenge report
```
@techreport{Moummad2023,
    Author = "Moummad, Ilyass and Serizel, Romain and Farrugia, Nicolas",
    title = "SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINING BIOACOUSTIC FEW SHOT SYSTEMS",
    institution = "DCASE2023 Challenge",
    year = "2023",
    month = "June",
}
```

### Or to cite the workshop paper version
```
@inproceedings{moummad,
    author = "Moummad, Ilyass and Serizel, Romain and Farrugia, Nicolas",
    title = "Pretraining Representations for Bioacoustic Few-Shot Detection Using Supervised Contrastive Learning",
    booktitle = "Proceedings of the 8th Detection and Classification of Acoustic Scenes and Events 2023 Workshop (DCASE2023)",
    address = "Tampere, Finland",
    month = "September",
    year = "2023",
    pages = "136--140",
}
```
