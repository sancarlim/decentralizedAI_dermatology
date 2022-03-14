# Federated Learning in Healthcare with Flower

One solution for issues with sharing health data to develop better models is decentralised AI, where a model is trained locally, and only model parameters get shared between sites. Sahlgrenska University Hospital team is currently involved in a large project on decentralised AI where we work together with AI Sweden and Region Halland (neighbouring healthcare providing area) to test this in practice.

In the first step of the project, we are working on publicly available data and simulating the decentralised structure internally. The main goal is to share learnings via presentation in AI Sweden and Information driven healthcare community. We have two use cases that we’re actively working on:

* Synthetic data generation using GANs with FL/SL setup.

* Image classification in FL/SL setup. 

Our main intrest is connected with Melanoma Image classification using more fair and accurate models.

## Skin Lesion Classification 

* Local Training in a centralized way: [`train_local.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/train_local.py)

* Training in a federated setup: 
  1. Launch [`server_advanced.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/server_advanced.py):    
 ```python server_advanced.py --path_data <> --r <Number of rounds for the federated training> --fc <Min fit clients, min number of clients to be sampled next round> --ac <Min available clients, min number of clients that need to connect to the server before training round can start>```
  The model is evaluated both centralized and in a decentralized manner. If you don’t want to perform centralized evaluation set `fraction_eval=0.0`.

  2. Launch one [`client_isic.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/client_isic.py) per terminal:
```python client_isic.py –path_data <path> –num_partitions <>  –partition  <> –gpu  <gpu ID>```

Note: Use `--nowandb` flag if you want to disable wandb logging.

## Synthetic data generation of skin lesions

We trained StyleGAN2-ADA in a federated setup (using Flower), simulating 3 hospitals (clients) with limited data, in order to achieve a more diverse, realistic and fair dataset. 

Our setup consists of 3 clients with different ISIC partitions based on patient ID (we made sure that data from one patient wasn’t distributed amongst different clients).
The clients have different amount of data: 2k, 10k, 20k respectively (as above Exp 2.).

For StylGAN2-ADA implementation we used:

* Local Training in a centralized way: [NVlabs/stylegan2-ada-pytorch](https://github.com/aidotse/stylegan2-ada-pytorch)

* Training in a federated setup: [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
  1. Launch [`server_advanced_gan.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/server_advanced_gan.py):    
 ```python server_advanced_gan.py --data <> --r <Number of rounds for the federated training> --fc <Min fit clients, min number of clients to be sampled next round> --ac <Min available clients, min number of clients that need to connect to the server before training round can start>```
  The model is evaluated in a decentralized manner. For some GANs parameters see the script.

  2. Launch one [`client_isic_gan.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/client_isic_gan.py) per terminal:
```python client_isic_gan.py –data <path> –num_partitions <>  –partition  <> –gpu  <gpu ID>```

Note: Use `--wandb` flag if you want to enable wandb logging.

###  Requirements

To use this FL GAN setup first clone [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch),
and install all dependences required for it with additional requirements for `flower framework`.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Flower Original README - A Friendly Federated Learning Framework

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/workflows/Build/badge.svg)
![Downloads](https://pepy.tech/badge/flwr)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://flower.dev/join-slack)

Flower (`flwr`) is a framework for building federated learning systems. The
design of Flower is based on a few guiding principles:

* **Customizable**: Federated learning systems vary wildly from one use case to
  another. Flower allows for a wide range of different configurations depending
  on the needs of each individual use case.

* **Extendable**: Flower originated from a research project at the Univerity of
  Oxford, so it was build with AI research in mind. Many components can be
  extended and overridden to build new state-of-the-art systems.

* **Framework-agnostic**: Different machine learning frameworks have different
  strengths. Flower can be used with any machine learning framework, for
  example, [PyTorch](https://pytorch.org),
  [TensorFlow](https://tensorflow.org), [Hugging Face Transformers](https://huggingface.co/), [PyTorch Lightning](https://pytorchlightning.ai/), [MXNet](https://mxnet.apache.org/), [scikit-learn](https://scikit-learn.org/), [TFLite](https://tensorflow.org/lite/), or even raw [NumPy](https://numpy.org/)
  for users who enjoy computing gradients by hand.

* **Understandable**: Flower is written with maintainability in mind. The
  community is encouraged to both read and contribute to the codebase.

Meet the Flower community on [flower.dev](https://flower.dev)!

## Documentation

[Flower Docs](https://flower.dev/docs):
* [Installation](https://flower.dev/docs/installation.html)
* [Quickstart (TensorFlow)](https://flower.dev/docs/quickstart_tensorflow.html)
* [Quickstart (PyTorch)](https://flower.dev/docs/quickstart_pytorch.html)
* [Quickstart (Hugging Face [code example])](https://flower.dev/docs/quickstart_huggingface.html)
* [Quickstart (PyTorch Lightning [code example])](https://flower.dev/docs/quickstart_pytorch_lightning.html)
* [Quickstart (MXNet)](https://flower.dev/docs/example-mxnet-walk-through.html)
* [Quickstart (scikit-learn [code example])](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)
* [Quickstart (TFLite on Android [code example])](https://github.com/adap/flower/tree/main/examples/android)

## Flower Usage Examples

A number of examples show different usage scenarios of Flower (in combination
with popular machine learning frameworks such as PyTorch or TensorFlow). To run
an example, first install the necessary extras:

[Usage Examples Documentation](https://flower.dev/docs/examples.html)

Quickstart examples:

* [Quickstart (TensorFlow)](https://github.com/adap/flower/tree/main/examples/quickstart_tensorflow)
* [Quickstart (PyTorch)](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch)
* [Quickstart (Hugging Face)](https://github.com/adap/flower/tree/main/examples/quickstart_huggingface)
* [Quickstart (PyTorch Lightning)](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch_lightning)
* [Quickstart (MXNet)](https://github.com/adap/flower/tree/main/examples/quickstart_mxnet)
* [Quickstart (scikit-learn)](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)
* [Quickstart (TFLite on Android)](https://github.com/adap/flower/tree/main/examples/android)

Other [examples](https://github.com/adap/flower/tree/main/examples):

* [Raspberry Pi & Nvidia Jetson Tutorial](https://github.com/adap/flower/tree/main/examples/embedded_devices)
* [Android & TFLite](https://github.com/adap/flower/tree/main/examples/android)
* [PyTorch: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/pytorch_from_centralized_to_federated)
* [MXNet: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/mxnet_from_centralized_to_federated)
* [Advanced Flower with TensorFlow/Keras](https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)
* [Single-Machine Simulation of Federated Learning Systems](https://github.com/adap/flower/tree/main/examples/simulation)

## Flower Baselines / Datasets

*Experimental* - curious minds can take a peek at [baselines](https://github.com/adap/flower/tree/main/baselines).

## Community

Flower is built by a wonderful community of researchers and engineers. [Join Slack](https://flower.dev/join-slack) to meet them, [contributions](#contributing-to-flower) are welcome.

<a href="https://github.com/adap/flower/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=adap/flower" />
</a>

## Citation

If you publish work that uses Flower, please cite Flower as follows: 

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

Please also consider adding your publication to the list of Flower-based publications in the docs, just open a Pull Request.

## Contributing to Flower

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) to get
started!
