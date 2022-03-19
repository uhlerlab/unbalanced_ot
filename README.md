# Scalable Unbalanced Optimal Transport with Generative Adversarial Networks

This is the accompanying code for the paper, "Scalable Unbalanced Optimal Transport with Generative Adversarial Networks" ([arXiv](https://arxiv.org/abs/1810.11447)). 

## Dependencies (Python 3)

* PyTorch
* NumPy
* Visdom

This code was developed on an NVIDIA GTX 1080TI GPU.

## Usage

To learn a transport map from source to target, run:

```
python main.py -sd results > results/log.txt
```

To obtain the transport results, run:
```
python eval.py -sd results 

```
The results will saved in `results`. By default, this code trains the model on the Zebrafish data shown in Figure 6. Our pretrained transport map can be downloaded from here: ([Google Drive](https://drive.google.com/file/d/1UjVreDsqoiqpDK-Z9VpNsp8E4yNxbTpB/view?usp=sharing)).
