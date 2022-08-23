python src/train.py -m logger=csv datamodule.transform_args.alpha=-0.5,-0.25,0,0.25,0.5 \
  datamodule.dataset=NCI1,NCI109,DD,PROTEINS,MUTAG,PTC_MR,ENZYMES,REDDIT-BINARY,REDDIT-MULTI-5K,COLLAB,IMDB-BINARY,IMDB-MULTI \
  logger=wandb logger.wandb.project=trainable_symmetry logger.wandb.tags=["v2"]
