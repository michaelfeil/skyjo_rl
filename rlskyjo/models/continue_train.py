from rlskyjo.models.train_model_simple_rllib import continual_train, init_ray
import os

if __name__ == "__main__":
    last_chpt_path = os.path.expanduser("~/")
    init_ray()
    continual_train(last_chpt_path, int(60*60*9.5))