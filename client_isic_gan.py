import os
from collections import OrderedDict
import numpy as np  
from typing import List, Tuple, Dict
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
from argparse import ArgumentParser 
import flwr as fl 
import utils_gans
from utils import seed_everything, seed_worker

import wandb 

import warnings

warnings.filterwarnings("ignore")
seed = 2022
seed_everything(seed)

EXCLUDE_LIST = [
    #"running",
    #"num_batches_tracked",
    #"bn",
]

class ClientGAN(fl.client.NumPyClient):
    """Flower client implementing melanoma generation using StyleGAN2-ADA."""

    def __init__(
        self,
        args,
        generator,
        discriminator,
        g_ema,
        g_optim, 
        d_optim,
        trainloader: torch.utils.data.DataLoader
    ) -> None:
        self.args = args
        self.generator = generator
        self.discriminator = discriminator
        self.g_ema = g_ema
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.trainloader = trainloader
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def get_properties(self, config):
        return {} 
    
    def get_parameters(self) -> List[np.ndarray]:
        g = [val.cpu().numpy() for _, val in self.generator.state_dict().items()]
        d = [val.cpu().numpy() for _, val in self.discriminator.state_dict().items()]
        gema = [val.cpu().numpy() for _, val in self.g_ema.state_dict().items()]
        # goptim = [val.numpy() for _, val in g_optim.state_dict().items()]
        # doptim = [val.numpy() for _, val in d_optim.state_dict().items()]
        model_weights = g + d + gema# + goptim + doptim

        return model_weights

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        len_gparam = len([val.cpu().numpy() for _, val in self.generator.state_dict().items()])
        len_dparam = len([val.cpu().numpy() for _, val in self.discriminator.state_dict().items()])
        len_emaparam = len([val.cpu().numpy() for _, val in self.g_ema.state_dict().items()])
        params_dict = zip(self.generator.state_dict().keys(), parameters[:len_gparam])
        gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        params_dict = zip(self.discriminator.state_dict().keys(), parameters[len_gparam:len_dparam+len_gparam])
        dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        params_dict = zip(self.g_ema.state_dict().keys(), parameters[-len_emaparam:])
        g_emastate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.generator.load_state_dict(gstate_dict, strict=False)
        self.discriminator.load_state_dict(dstate_dict, strict=False)
        self.g_ema.load_state_dict(g_emastate_dict, strict=False)
    

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        self.generator, self.discriminator, self.g_ema = utils_gans.train_gan(
            self.args, self.trainloader, self.generator,
            self.discriminator, self.g_ema,
            self.g_optim, self.d_optim,
            self.device, self.args.partition)
        return self.get_parameters(), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, Dict]: 
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        save_path=f"/workspace/data/sample/testa_client"
        im, caption = utils_gans.val_gan(self.args, self.g_ema, self.generator,
                                         self.discriminator,
                                         self.g_optim, self.d_optim,
                                         i=args.partition,
                                         clin_id=args.partition,
                                         save_path=save_path)

        if self.args.wandb:
            wandb.log({"current grid": wandb.Image(im, caption=caption)})

        torch.save(
                    {
                        "g": self.generator.state_dict(),
                        "d": self.discriminator.state_dict(),
                        "g_ema": self.g_ema.state_dict(),
                        "g_optim": self.g_optim.state_dict(),
                        "d_optim": self.d_optim.state_dict(),
                        "args": self.args,
                    },
                    f"{save_path}-{str(args.partition)}/best_{str(0).zfill(6)}.pt",
                )

        return float(1), {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='/workspace/data/data/melanoma_external_256')
    parser.add_argument("--num_partitions", type=int, default=3) 
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--iter", type=int, default=100, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training", )
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization", )
    parser.add_argument("--path_batch_shrink", type=int, default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)", )
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization", )
    parser.add_argument("--g_reg_every", type=int, default=4,
        help="interval of the applying path length regularization", )
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training", )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1", )
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation", )
    parser.add_argument("--ada_target", type=float, default=0.6,
        help="target augmentation probability for adaptive augmentation", )
    parser.add_argument("--ada_length", type=int, default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation", )
    parser.add_argument("--ada_every", type=int, default=256,
        help="probability update interval of the adaptive augmentation", )
    parser.add_argument("--num_classes", type=int, default=2, )
    parser.add_argument("--gpu", type=int, default=0, )
    parser.add_argument("--distributed", type=int, default=0, )
    parser.add_argument("--latent", type=int, default=512, )
    parser.add_argument("--n_mlp", type=int, default=8, )
    parser.add_argument("--start_iter", type=int, default=0, )
    parser.add_argument('--decimate', type=int, default=1, help='select decimation modality for stylegan dataloader')
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    args = parser.parse_args()


    # Setting up GPU for processing or CPU if GPU isn't available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load model
    generator, discriminator, g_ema, g_optim, d_optim = utils_gans.load_ganmodel(args)

    if args.wandb:
        wandb.init(project="dai-healthcare",
                group="FL_GANs",
                entity='eyeforai',
                config={"model": args.arch})
        wandb.config.update(args)

    # Load data
    # Normal partition
    trainset, valset, num_examples = utils_gans.load_isic_gandata(data_path=args.data)
    trainset, valset, num_examples = utils_gans.load_isic_by_patient(args.partition, args.path)
    # Exp 1
    # trainset, testset, num_examples = utils.load_exp1_partition(trainset, testset, num_examples, idx=args.partition)
    # Exp 2-6
    # train_df, validation_df, num_examples = utils.load_isic_by_patient(args.partition, args.path)
    # trainset = utils_gans.CustomDataset(df=train_df, train=True, transforms=training_transforms) 
    # valset = utils_gans.CustomDataset(df=validation_df, train=True, transforms=testing_transforms)

    print(f"Train dataset: {len(trainset)}, Val dataset: {len(valset)}")

    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, worker_init_fn=seed_worker, shuffle=True)

    # Start client 
    client = ClientGAN(args, generator, discriminator, g_ema, g_optim, d_optim, train_loader)
    fl.client.start_numpy_client("0.0.0.0:8080", client)
