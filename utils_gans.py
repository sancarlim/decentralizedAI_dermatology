import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch

from torch import optim
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append('/workspace/stylegan2-pytorch')
from sequencedataloader import txt_dataloader_styleGAN
from torchvision import transforms, utils
import random


# Creating seeds to make results reproducible
def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 2022
seed_everything(seed)

# Handle data
gantransform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )


def load_isic_gandata(data_path='/workspace/data/data/melanoma_external_256/'):
    # ISIC Dataset
    df = pd.read_csv(os.path.join(data_path, "train_concat.csv"))
    train_img_dir = os.path.join(data_path, "train/train/")

    df['image_name'] = [os.path.join(
        train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))]
    train_split, valid_split = train_test_split(
        df, stratify=df.target, test_size=0.20, random_state=42) 
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split) 

    training_dataset = CustomGanDataset(df=train_df, transform=gantransform) 
    testing_dataset = CustomGanDataset(df=validation_df, transform=gantransform) 

    num_examples = {"trainset":len(training_dataset), "testset":len(testing_dataset)} 

    return training_dataset, testing_dataset, num_examples


def load_isic_by_patient_client(
    partition,
    data_path='/workspace/data/data/melanoma_external_256'
    ):
    # Load data
    df = pd.read_csv(os.path.join(data_path, "train_concat.csv"))
    train_img_dir = os.path.join(data_path, "train/train/")

    df['image_name'] = [os.path.join(
        train_img_dir, df.iloc[index]['image_name'] + '.jpg'
        ) for index in range(len(df))]
    df["patient_id"] = df["patient_id"].fillna('nan')

    # Split by Patient 
    patient_groups = df.groupby('patient_id') #37311
    melanoma_groups_list = [
        patient_groups.get_group(x) for x in patient_groups.groups if patient_groups.get_group(x)['target'].unique().all()==1]  # 4188 - after adding na 4525
    benign_groups_list = [
        patient_groups.get_group(x) for x in patient_groups.groups if 0 in patient_groups.get_group(x)['target'].unique()]  # 2055 - 33123

    np.random.shuffle(melanoma_groups_list)
    np.random.shuffle(benign_groups_list)

    if partition == 2:
        df_b_test = pd.concat(benign_groups_list[1800:]) # 4462 
        df_b_train = pd.concat(benign_groups_list[800:1800])  # 16033 - TOTAL 20495 samples 
        df_m_test = pd.concat(melanoma_groups_list[170:281]) # 340  
        df_m_train = pd.concat(melanoma_groups_list[281:800])  # 1970 - TOTAL: 2310 samples 
    elif partition == 1:
        df_b_test = pd.concat(benign_groups_list[130:250])  #  1949  
        df_b_train = pd.concat(benign_groups_list[250:800])  # 8609 - TOTAL 10558 samples  
        df_m_test = pd.concat(melanoma_groups_list[1230:]) # 303 
        df_m_train = pd.concat(melanoma_groups_list[800:1230]) # 1407 - TOTAL 1710 samples  
    else:
        df_b_test = pd.concat(benign_groups_list[:30])  # 519
        df_b_train = pd.concat(benign_groups_list[30:130]) # 1551 - TOTAL: 2070 samples 
        df_m_test = pd.concat(melanoma_groups_list[:70])  # 191
        df_m_train = pd.concat(melanoma_groups_list[70:170]) # 314 - TOTAL: 505 samples

    train_split = pd.concat([df_b_train, df_m_train])
    valid_split = pd.concat([df_b_test, df_m_test])

    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split) 

    training_dataset = CustomGanDataset(df=train_df, transform=gantransform) 
    testing_dataset = CustomGanDataset(df=validation_df, transform=gantransform)

    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 

    return training_dataset, testing_dataset, num_examples


class CustomGanDataset(txt_dataloader_styleGAN):
    def __init__(self, df: pd.DataFrame, conditional=True, transform=None,
                 usePIL=True, isSequence=False, GANflag = False, verbose=True):
        self.df = df
        self.transform = transform
        self.conditional = conditional
        self.GANflag = GANflag
        self.verbose = verbose
        self.images = list(df["image_name"])
        self.labels = list(df["target"].values)
        if self.verbose:
            print('Images loaded: ' + str(len(self.images)) + '\n')
        self.usePIL = usePIL
        self.isSequence = isSequence
        self.imgs = list(zip(self.images, self.labels))


def load_experiment_partition2(trainset, testset, num_examples, idx):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert idx in range(3)  

    if idx==0:
        train_partition = torch.utils.data.Subset(
            trainset, range(0, 2000)
        )
        test_partition = torch.utils.data.Subset(
            testset, range(0,502)
        )
        print('Train loaded: ' + str(len(train_partition)) + '\n')
        print('Test loaded: ' + str(len(test_partition)) + '\n')

        train_partition, test_partition = trainset.images[0:2000], testset.images[0:502]
    
    elif idx==1: 
        train_partition = torch.utils.data.Subset(
            trainset, range(5000, 10000)
        )
        test_partition = torch.utils.data.Subset(
            testset, range(600, 1855)
        )
        print('Train loaded: ' + str(len(train_partition)) + '\n')
        print('Test loaded: ' + str(len(test_partition)) + '\n')

        train_partition = trainset.images[5000:10000]
        test_partition = testset.images[600:1855]
    else:
        train_partition = torch.utils.data.Subset(
            trainset, range(10000, 20000)
        )
        test_partition = torch.utils.data.Subset(
            testset, range(2000, 4510)
        )
        print('Train loaded: ' + str(len(train_partition)) + '\n')
        print('Test loaded: ' + str(len(test_partition)) + '\n')

        train_partition = trainset.images[10000:20000]
        test_partition = testset.images[2000:4510]

    num_examples = {"trainset" : len(train_partition), "testset" : len(test_partition)} 

    return (train_partition, test_partition, num_examples)


# Handle model
def load_ganmodel(args):
    if args.arch == 'stylegan2':
        from model_conditional import Generator, Discriminator
    elif args.arch == 'swagan':
        from swagan_conditional import Generator, Discriminator
    from train_conditional import accumulate
    generator = Generator(
        args.size, args.latent, args.n_mlp,
        num_classes=args.num_classes,
        channel_multiplier=args.channel_multiplier
    )
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier,
        num_classes=args.num_classes
    )
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, num_classes=args.num_classes, channel_multiplier=args.channel_multiplier
    )
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    return generator, discriminator, g_ema, g_optim, d_optim


def train_gan(args, loader, generator,
              discriminator, g_ema,
              g_optim, d_optim,
              device, id):

    from train_conditional import train
    # Training model
    generator.to(device)
    discriminator.to(device)
    g_ema.to(device)

    print('Starts training...')
    return train(args, loader, generator, discriminator,
                 g_optim, d_optim, g_ema,
                 device, args.partition, id)


def val_gan(args, g_ema, g_module, d_module,
            g_optim, d_optim, i, clin_id=0,
            save_path="/workspace/data/sample/test_client"):
    g_ema.to(args.device)
    g_ema.eval()
    sample_z = torch.randn(args.n_sample, args.latent, device=args.device)
    sample_labels = torch.tensor([0,1]).repeat(args.n_sample//args.num_classes)
    sample_labels = torch.nn.functional.one_hot(sample_labels, num_classes=args.num_classes).float().to(args.device)

    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        sample, _ = g_ema([sample_z], sample_labels)
        utils.save_image(
            sample,
            f"{save_path}-{str(clin_id)}/{str(i).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(-1, 1),
        )
        grid = utils.make_grid(sample, nrow=int(
            args.n_sample ** 0.5),normalize=True, range=(-1, 1))
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy().astype(np.float32)
        im = ndarr

        torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"{save_path}-{str(clin_id)}/{str(i).zfill(6)}.pt",
                )

    return im, f"Iter:{str(i).zfill(6)}"
