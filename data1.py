import syft as sy
import torch 
import numpy as np
import pandas as pd
import os, sys

sys.path.append('/workspace/stylegan2-ada-pytorch')

from melanoma_cnn_efficientnet import CustomDataset, seed_everything, train_test_split, training_transforms, testing_transforms, create_split

# Launch a Duet server 
duet = sy.launch_duet(loopback=True)


# Create ISIC dataset
df = pd.read_csv('/workspace/melanoma_isic_dataset/train_concat.csv') 
train_img_dir = '/workspace/melanoma_isic_dataset/train/train/'
df['image_name'] = [os.path.join(train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))] 
train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 

train_df=pd.DataFrame(train_split).tag('isic_train_data')
validation_df=pd.DataFrame(valid_split).tag('isic_val_data')

# Data owner adds a description to the tensor where data is located
train_df = train_df.describe("This is ISIC2020 training data.")

# Finally the data owner UPLOADS THE DATA to the Duet server and makes it searchable
# by data scientists. NOTE: The data is still on the Data Owners machine and cannot be
# viewed or retrieved by any Data Scientists without permission.
data_train_pointer = train_df.send(duet, pointable=True)
data_val_pointer = validation_df.send(duet, pointable=True)

# Once uploaded, the data owner can see the object stored in the tensor
duet.store

# To see it in a human-readable format, data owner can also pretty-print the tensor information
duet.store.pandas

# To check if there is a request from the Data Scientist, the data owner runs this command occasionally
# or when there is a notification of new request in the DUET LIVE STATUS
duet.requests.pandas
# The request looks reasonable. Should be accepted :)
duet.requests[0].accept()

# You can automatically accept or deny requests, which is great for testing.
# We have more advanced handlers coming soon.
duet.requests.add_handler(action="accept")