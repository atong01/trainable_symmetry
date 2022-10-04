python src/train.py -m datamodule.transform_args.alpha=-0.5,-0.25,0.0,0.25,0.5 \
datamodule.dataset=NCI1,NCI109,DD,PROTEINS,MUTAG,PTC_MR,ENZYMES,REDDIT-BINARY,REDDIT-MULTI-5K,COLLAB,IMDB-BINARY,IMDB-MULTI \
  logger=wandb logger.wandb.project=trainable_symmetry logger.wandb.tags=["v6","lr","power1"] datamodule.transform_args.power=1,2\
  seed=0,1,2,3,4,5,6,7,8,9

python src/train.py -m datamodule.transform_args.alpha=-0.5,-0.25,0.0,0.25,0.5 \
datamodule.dataset=NCI1,NCI109,DD,PROTEINS,MUTAG,PTC_MR,ENZYMES,REDDIT-BINARY,REDDIT-MULTI-5K,COLLAB,IMDB-BINARY,IMDB-MULTI \
  logger=wandb logger.wandb.project=trainable_symmetry logger.wandb.tags=["v6","lr","power1"] datamodule.transform_args.power=1 +datamodule.transform_args.cheb_order=10,100\
  seed=0,1,2,3,4,5,6,7,8,9
