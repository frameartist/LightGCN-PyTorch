import world
import dataloader
#import dataloader_ratings as dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml_latest_small':
    dataset = dataloader.MovieLens()
elif world.dataset == 'indonesia_tourism':
    dataset = dataloader.MovieLens(path="../data/indonesia_tourism/",filename="tourism_rating.csv", names=["User_Id","Place_Id","Place_Ratings"],columns=["User_Id","Place_Id","Place_Ratings"])
    
print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'lmse': model.LightGCNMSE
}