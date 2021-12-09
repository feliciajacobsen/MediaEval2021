# MediaEval2021
Source code, dockerimage, dockerfile to the MediaEval2021 Challenge of the subtask: Transparency Polyp Segmentation. This project was made using the PyTorch framework. Model weights (.pt-files) are not uploaded due to Github memory constraints. I used Deep Ensembles (https://arxiv.org/pdf/1612.01474v3.pdf) as the mehtod of predicting segmentation masks on polyps based on colonoscopy images of the GI tract, and to obtain uncertainty values of each prediction mask based on the variance of the different models in the ensemble. 

## Sofware requirements
All Python libraries are shown in requirements.txt.

## How to Run
By default, the code will use cuda (assuming that it is available). However, if GPU is not available, change the variable `config["use_gpu"]` to `False`.

To run everything, execute:

```
$ python test.py
```

test.py simply controls the code which is in the /src directory.
