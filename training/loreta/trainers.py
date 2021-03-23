"""Training logic for logo recognition."""

# Pip packages:
from tqdm import tqdm
import numpy as np
import torch

# Local:
from .pytorch import detect_utils
from .pytorch.coco_utils import get_coco_api_from_dataset
from .pytorch.coco_eval import CocoEvaluator


class Trainer(object):
    def __init__(self, config):
        # Data:
        self.loader_train = config["loader_train"]
        self.loader_valid = config["loader_valid"]
        self.logger = config["logger"]

        # Training:
        self.n_epochs = config["general"]["n_epochs"]
        self.early_stop = config["general"]["early_stop"]
        self.criterion = config["criterion"]
        self.min_loss = np.inf
        self.last_loss = np.inf
        self.stop_counter = 0

        # Model:
        self.model = config["model"]
        self.device = torch.device(config["general"]["device"])
        self.model.to(self.device)
        self.optimizer = config["optimizer"]

        # Seed:
        self.seed = config["general"]["seed"]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def train(self):
        n_train_batches = len(self.loader_train)
        n_valid_batches = len(self.loader_valid)

        for epoch in range(self.n_epochs):
            print("\nEpoch {} of {}:".format(epoch + 1, self.n_epochs))

            # Compute losses:
            train_loss, train_acc = self._train_epoch(
                n_train_batches, self.loader_train
            )
            valid_loss, valid_acc = self._valid_epoch(
                n_valid_batches, self.loader_valid
            )

            # Log results:
            self.logger.log(
                train_loss, valid_loss, self.model, self.optimizer, train_acc, valid_acc
            )
            if self.check_stop():
                break

    def check_stop(self):
        done = False

        # If performance improves:
        if self.last_loss < self.min_loss:
            self.min_loss = self.last_loss
            # Set early-stop counter to 0:
            self.stop_counter = 0
        else:
            self.stop_counter += 1

        # If performance stagnates:
        if self.stop_counter == self.early_stop:
            n_epochs = "No improvement for {} epochs.".format(self.early_stop)
            print("{} Training stopped.".format(n_epochs))
            done = True

        return done

    def _train_epoch(self, n_batches, data_loader):
        raise NotImplementedError

    def _valid_epoch(self, n_batches, data_loader):
        raise NotImplementedError


class ClassifierTrainer(Trainer):
    def __init__(self, config):
        Trainer.__init__(self, config)

    def _train_epoch(self, n_batches, data_loader):
        epoch_loss = 0.0
        epoch_acc = 0.0

        self.model.train()
        for image, target in tqdm(data_loader, desc="Training", total=n_batches):
            # Forward pass:
            loss, acc = self._forward_batch(image, target)
            # Logs:
            epoch_loss += loss.item()
            epoch_acc += acc
            # Back-prop pass:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= n_batches
        epoch_acc /= n_batches
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _valid_epoch(self, n_batches, data_loader):
        epoch_loss = 0.0
        epoch_acc = 0.0

        self.model.eval()
        for image, target in tqdm(data_loader, desc="Validating", total=n_batches):
            # Forward pass:
            loss, acc = self._forward_batch(image, target)
            # Logs:
            epoch_loss += loss.item()
            epoch_acc += acc

        epoch_loss /= n_batches
        epoch_acc /= n_batches
        self.last_loss = epoch_loss
        return epoch_loss, epoch_acc

    def _forward_batch(self, images, targets):
        # Get model output:
        images, targets = images.to(self.device), targets.to(self.device)
        output = self.model(images)
        # Calculate loss:
        loss = self.criterion(output.squeeze(), targets)
        # Calculate accuracy:
        if output.shape[1] > 1:
            pred_np = np.argmax(output.cpu().detach().numpy(), 1)
        else:
            pred_np = np.round(torch.sigmoid(output).cpu().detach().numpy())
        target_np = targets.cpu().detach().numpy()
        acc = np.mean((pred_np.squeeze() == target_np).astype(np.float32))
        return loss, acc


class DetectorTrainer(Trainer):
    def __init__(self, config):
        Trainer.__init__(self, config)

        # Build evaluator:
        coco = get_coco_api_from_dataset(self.loader_valid.dataset)
        iou_types = detect_utils._get_iou_types(self.model)
        self.evaluator = CocoEvaluator(coco, iou_types)

    def _train_epoch(self, n_batches, data_loader):
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for images, targets in tqdm(data_loader, desc="Training", total=n_batches):

            # Move data to GPU:
            images, targets = self.move_to_gpu(images, targets)
            # Forward pass:
            loss_dict = self.model(images, targets)
            # Calculate loss:
            loss = sum(loss for loss in loss_dict.values())
            # Logs:
            epoch_loss += loss
            # Back-prop pass:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= n_batches
        self.last_loss = epoch_loss
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _valid_epoch(self, n_batches, data_loader):
        self.model.eval()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for images, targets in tqdm(data_loader, desc="Validating", total=n_batches):

            # Move data to GPU:
            images, targets = self.move_to_gpu(images, targets)
            # Forward pass:
            torch.cuda.synchronize()
            outputs = self.model(images)
            outputs = [{k: v.to(self.device) for k, v in o.items()} for o in outputs]
            results = {
                target["image_id"].item(): output
                for target, output in zip(targets, outputs)
            }
            self.evaluator.update(results)

        # Gather the stats from all processes:
        self.evaluator.synchronize_between_processes()
        self.evaluator.accumulate()
        self.evaluator.summarize()

        eval_stats = self.evaluator.coco_eval["bbox"].stats
        if len(eval_stats) == 0:
            min_precision = min(eval_stats[:6])
            min_recall = min(eval_stats[6:])
            epoch_acc = min_precision + min_recall  # Substitute reference.

        return epoch_loss, epoch_acc

    def move_to_gpu(self, images, targets):
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets
