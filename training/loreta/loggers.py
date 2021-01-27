"""Data loggers for monitoring experiments and saving models."""

# Standard library:
import os
import json
import time
from datetime import datetime

# Pip packages:
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TBLogger(object):
    def __init__(self, config_file):

        # Set the save dir:
        self.root = self._set_root(config_file)
        print("\n[INFO] Saving in {}\n".format(self.root))

        # Create subfolder for models:
        self.model_path = os.path.join(self.root, "models")
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # Initialize a TensorBoard writer there:
        self.writer = SummaryWriter(self.root)

        # Save the entire JSON there
        file_handle = open(os.path.join(self.root, "config.json"), "w")
        json.dump(config_file, file_handle, indent=4)

        # Housekeeping:
        self.save_every = config_file["logger"]["args"]["save_every"]
        self.epoch = 0
        self.best_valid_loss = np.inf

    def _set_root(self, config_file):
        if "__auto__" in config_file["logger"]["args"]["root"]:
            # Set a time stamp:
            time_stamp = datetime.fromtimestamp(time.time())
            time_stamp = time_stamp.strftime("date_%y%m%d_time_%H%M%S")

            # Give it a catchy name:
            log_name = "{}_".format(config_file["model"]["type"])
            for key, val in config_file["model"]["args"].items():
                log_name += "{}_{}_".format(key, val)

            log_name += "{}_".format(config_file["optimizer"]["type"])
            for key, val in config_file["optimizer"]["args"].items():
                log_name += "{}_{}_".format(key, val)

            log_name += "{}_{}_{}_{}"
            log_name = log_name.format(
                config_file["general"]["batchsize"],
                config_file["transform"]["train"]["type"],
                config_file["general"]["seed"],
                time_stamp,
            )
            root = config_file["logger"]["args"]["root"].replace("__auto__", log_name)
        else:
            root = config_file["logger"]["args"]["root"]
        return root

    def save(
        self,
        train_loss,
        valid_loss,
        model=None,
        optimizer=None,
        train_acc=None,
        valid_acc=None,
        save_add="",
    ):
        checkpoint = {
            "model": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": valid_loss,
            "optimizer": optimizer.state_dict(),
            "epoch": self.epoch,
        }
        if train_acc is not None:
            checkpoint["train_acc"] = train_acc
        if valid_acc is not None:
            checkpoint["valid_acc"] = valid_acc

        # Define save path:
        if save_add == "":
            model_name = "checkpoint_{:04d}.pth".format(self.epoch)
        if save_add == "best":
            model_name = "checkpoint_best.pth"
            # To quickly know which was the best epoch
            filename = os.path.join(
                self.model_path, "Best model epoch {}".format(self.epoch)
            )
            open(filename, "w").close()
        else:
            model_name = "checkpoint_{:04d}_{}.pth".format(self.epoch, save_add)
        path = os.path.join(self.model_path, model_name)
        torch.save(checkpoint, path)

    def log(
        self, train_loss, valid_loss, model, optimizer, train_acc=None, valid_acc=None
    ):
        self.epoch += 1

        # Add logs:
        self.writer.add_scalar("Loss/train", train_loss, self.epoch)
        self.writer.add_scalar("Loss/valid", valid_loss, self.epoch)
        if train_acc is not None:
            self.writer.add_scalar("Accuracy/train", train_acc, self.epoch)
        if valid_acc is not None:
            self.writer.add_scalar("Accuracy/valid", valid_acc, self.epoch)
        self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], self.epoch)

        # Save data:
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.save(
                train_loss,
                valid_loss,
                model,
                optimizer,
                train_acc,
                valid_acc,
                save_add="best",
            )
        if self.epoch % self.save_every == 0:
            self.save(train_loss, valid_loss, model, optimizer, train_acc, valid_acc)
