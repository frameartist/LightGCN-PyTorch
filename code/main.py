import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import pandas as pd
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
mse = utils.MSELoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        #output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        output_information = Procedure.MSE_train_original(dataset, Recmodel, mse, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
    from sklearn.metrics import mean_squared_error 
    (users_emb, items_emb, 
        userEmb0,  itemsEmb0) = Recmodel.getEmbedding(torch.Tensor(dataset.testUser).long(), torch.Tensor(dataset.testItem).long())
    item_scores = torch.mul(users_emb, items_emb)
    item_scores = torch.sum(item_scores, dim=1)
    predictions = np.clip(item_scores.detach().numpy(), 0, 5)
    print("RMSE",mean_squared_error(predictions, dataset.testRating, squared=False))
    df = pd.DataFrame()
    df['user'] = dataset.testUser
    df['item'] = dataset.testItem
    df['actual'] = dataset.testRating
    df['predicted'] = predictions
    df.to_csv("ml_result.csv")
finally:
    if world.tensorboard:
        w.close()