# Flower Example using PyTorch and CIFAR10

[Flower Quickstart Tutorial](https://flower.dev/docs/quickstart_pytorch.html)

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.
This example consists of one server and two clients all having the same model. 

The included [`run.sh`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/run.sh) will start the Flower server (using `server.py`), sleep for 5 seconds to ensure the the server is up, and then start 3 Flower clients (using `client_cifar.py`). You can simply start everything in a terminal as follows:

```sh ./run.sh```


## Advanced Example with Pytorch

[`client_advanced_cifar.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/advanced_example/server_advanced_cifar.py) and [`server_advanced_cifar.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/advanced_example/server_advanced_cifar.py)  includes a more advanced pytorch tutorial with the following additions:
* Additional configurations for the server
* Server-side model evaluation after parameter aggregation 
* Hyperparameter schedule using config functions
* Custom return values
* Server-side parameter initialization

The included [`run.sh`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/advanced_example/run.sh) will start the Flower server (using `server_advanced_cifar.py`), sleep for 5 seconds to ensure the the server is up, and then start 3 Flower clients (using `client_advanced_cifar.py`). You can simply start everything in a terminal as follows:

```sh ./run.sh <number of clients> <Nb min of clients before launching round> <Nb of clients sampled for the round> <Nb of rounds>```

The `run.sh` script starts processes in the background so that you don't have to open N terminal windows (N-1 clients). If you experiment with the code example and something goes wrong, simply using CTRL + C on Linux (or CMD + C on macOS) wouldn't normally kill all these processes, which is why the script ends with trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT and wait. This simply allows you to stop the experiment using CTRL + C (or CMD + C). If you change the script and anything goes wrong you can still use killall python (or killall python3) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).

## Scaling Flower with Multiprocessing

If you want to emulate a federated learning setting locally, it’s really easy to scale to as much clients as your CPU allows you to. For basic models, CPU is more than enough and there is no need to extend training on GPU. However when using bigger models or bigger datasets, you might want to move to GPU in order to greatly improve the training speed.  This is where you can encounter a problem in scaling your Federated setting.
Indeed, unlike some other frameworks, Flower’s goal is to allow easy deployment from research/prototype to production so they treat clients as independent processes. Additionally, when accessing the GPU, CUDA will automatically allocate a fixed amount of memory so that it has enough room to work with before asking for more.
However, this memory can’t be freed at all, at least not until the process exits. This means that if you are launching 100 clients and sample 10 of them per round and are using the GPU, every time a client will access it, there will be leftover memory that can’t be released and it will keep growing as new clients are sampled. In the long term, your GPU needs as much memory as there are clients launched.

Since the memory is not released until the process accessing it is released, then we simply need to encapsulate the part of our code that need to access the GPU in a sub-process, waiting for it to be terminated until we can continue to execute our program. Multiprocessing is the solution, and I will show you how to do it using PyTorch and Flower.

 [`client_cifar_mp.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/multiprocessing/client_cifar_mp.py) and  [`server_mp.py`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/multiprocessing/server_mp.py) implement the solution proposed [here](https://towardsdatascience.com/scaling-flower-with-multiprocessing-a0bc7b7aace0#a341-919044032aa0).
 
 The included [`run.sh`](https://github.com/aidotse/decentralizedAI_dermatology/blob/master/cifar/multiprocessing/run.sh) will start the Flower server (using `server_mp.py`), sleep for 5 seconds to ensure the the server is up, and then start 3 Flower clients (using `client_cifar_mp.py`). You can simply start everything in a terminal as follows:
 
```sh ./run.sh <number of clients> <Nb min of clients before launching round> <Nb of clients sampled for the round> <Nb of rounds>```
