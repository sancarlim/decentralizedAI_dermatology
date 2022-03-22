import sys
sys.path.append('/workspace/flower')
import src.py.flwr as fl 
from typing import List, Tuple, Dict, Optional
import sys, os
import numpy as np
import torch
from collections import OrderedDict
import utils_gans, utils
import warnings
import wandb
from argparse import ArgumentParser  

warnings.filterwarnings("ignore")

# Setting up GPU for processing or CPU if GPU isn't available
#device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

seed = 2022
utils.seed_everything(seed)

def set_parameters(generator, discriminator, g_ema, parameters: List[np.ndarray]) -> None:
    # Set model parameters from a list of NumPy ndarrays
    len_gparam = len([val.cpu().numpy() for _, val in generator.state_dict().items()])
    len_dparam = len([val.cpu().numpy() for _, val in discriminator.state_dict().items()])
    len_emaparam = len([val.cpu().numpy() for _, val in g_ema.state_dict().items()])
    #len_doptim = len([val.cpu().numpy() for _, val in d_optim.state_dict().items()])
    params_dict = zip(generator.state_dict().keys(), parameters[:len_gparam])
    gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    params_dict = zip(discriminator.state_dict().keys(), parameters[len_gparam:len_dparam+len_gparam])
    dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    params_dict = zip(g_ema.state_dict().keys(), parameters[-len_emaparam:])
    g_emastate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #params_dict = zip(g_ema.state_dict().keys(), parameters[len_dparam+len_gparam+len_emaparam:-len_doptim])
    #goptstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #params_dict = zip(g_ema.state_dict().keys(), parameters[-len_doptim:])
    #doptstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    generator.load_state_dict(gstate_dict, strict=False)
    discriminator.load_state_dict(dstate_dict, strict=False)
    g_ema.load_state_dict(g_emastate_dict, strict=False)
    #g_optim.load_state_dict(goptstate_dict, strict=False)
    #d_optim.load_state_dict(doptstate_dict, strict=False)

def get_eval_fn(generator, discriminator, g_ema, g_optim, d_optim, args):
    """Return an evaluation function for server-side evaluation."""
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        set_parameters(generator, discriminator, g_ema, weights)
        save_path="/workspace/data/sample/testa_client"
        im, caption = utils_gans.val_gan(args, g_ema, generator, discriminator,
                                         g_optim, d_optim, i=-1, clin_id=-1,
                                         save_path=save_path)

        if args.wandb:
            wandb.log({"current grid": wandb.Image(im, caption=caption)})

        torch.save(
                    {
                        "g": generator.state_dict(),
                        "d": discriminator.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"{save_path}-{str(-1)}/best2_{str(0).zfill(6)}.pt",
                )

        return float(1), {}
    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "rnd": rnd,
        "batch_size": 16,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"rnd": rnd, "val_steps": val_steps}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='/workspace/data/data/melanoma_external_256')
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-fc",
        type=int,
        default=3,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=3,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
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

    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    # Setting up GPU for processing or CPU if GPU isn't available
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # Load model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation
    generator, discriminator, g_ema, g_optim, d_optim = utils_gans.load_ganmodel(args)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    g = [val.cpu().numpy() for _, val in generator.state_dict().items()]
    d = [val.cpu().numpy() for _, val in discriminator.state_dict().items()]
    gema = [val.cpu().numpy() for _, val in g_ema.state_dict().items()]
    # goptim = [val.numpy() for _, val in g_optim.state_dict().items()]
    # doptim = [val.numpy() for _, val in d_optim.state_dict().items()]
    model_weights = g + d + gema# + goptim + doptim

    if args.wandb:
        wandb.init(project="dai-healthcare",
                group="FL_GANs",
                entity='eyeforai',
                config={"model": args.arch})
        wandb.config.update(args)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = fc/ac,
        fraction_eval = 0.2, # not used - no federated evaluation
        min_fit_clients = fc,
        min_eval_clients = 2, # not used 
        min_available_clients = ac,
        eval_fn=get_eval_fn(generator, discriminator, g_ema, g_optim, d_optim, args),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model_weights), 
    )

    fl.server.start_server("0.0.0.0:8081", config={"num_rounds": rounds}, strategy=strategy)
