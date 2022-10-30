import wandb
WANDB_DIR="/home/orange/Main/Experiment/ICLR/cifar10/vgg16_express_0"
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run(r"express_0/2ismlwao")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")