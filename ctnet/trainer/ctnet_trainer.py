import time
from decimal import Decimal
from torch.nn import BCELoss
import datetime
import numpy as np
from tqdm import tqdm
import os
from gnutools.fs import parent
import torch
from ctnet.trainer import f1_score


class DefaultTrainer:
    def __init__(
            self,
            model,
            optimizer,
            optimizer_kwargs,
            criterion,
            train_loader,
            dev_loader,
            model_path,
            checkpoint_path,
            epochs,
            fit_epoch,
            eval_epoch,
            checkpoint_epoch=None,
            sync_epoch=None):
        self.model=model
        self.optimizer=optimizer(model.parameters(),**optimizer_kwargs)
        self.criterion=criterion
        self.train_loader=train_loader
        self.dev_loader=dev_loader
        self.model_path=model_path
        self.checkpoint_path=checkpoint_path
        self.epochs=epochs
        self.num_workers=32
        self.epoch=0
        self.fit_epoch=fit_epoch
        self.eval_epoch=eval_epoch
        self.checkpoint_epoch=checkpoint_epoch
        self.sync_epoch=sync_epoch
        self.losses = {
            "history": {
                "train": [],
                "dev": []
            },
            "avg": {
                "train": None,
                "dev": None
            },
            "best": {
                "train": None,
                "dev": None
            }

        }

        self.f1_score = {
            "history": {
                "train": [],
                "dev": []
            },
            "avg": {
                "train": None,
                "dev": None
            },
            "best": {
                "train": None,
                "dev": None
            }

        }


        self.time_epoch=None
        self.last_update=None

    def desc(self):
        def get_estimate(duration, epoch, epochs):
            if duration is None:
                return None
            else:
                estimate = (epochs - epoch) * duration
                return datetime.timedelta(seconds=estimate)

        return "{name} >> {epoch}/{epochs} | " \
               "LU: {last_update} | " \
               "Time: {duration}s/{estimated} | " \
               "Lr: {lr} | " \
               "Loss: {loss_train}/{loss_dev}({loss_dev_best}) | " \
               "F1: {f1_train}/{f1_dev}({f1_dev_best}) | " \
               "Plt: {plateau}".format(
            name=self.model.id,
            epoch=self.epoch,
            epochs=self.epochs,
            duration=self.time_epoch,
            estimated=get_estimate(self.time_epoch,
                                   self.epoch,
                                   self.epochs),
            last_update=self.last_update,
            lr='%.2E' % Decimal(self.optimizer.defaults['lr']),
            #F1
            f1_train='%.2E' % Decimal(self.f1_score["avg"]["train"]) if self.f1_score["avg"]["train"] is not None else None,
            f1_dev='%.2E' % Decimal(self.f1_score["avg"]["dev"]) if self.f1_score["avg"]["dev"] is not None else None,
            f1_dev_best='%.2E' % Decimal(self.f1_score["best"]["dev"]) if self.f1_score["best"]["dev"] is not None else None,
            #Loss
            loss_train='%.2E' % Decimal(self.losses["avg"]["train"]) if self.losses["avg"]["train"] is not None else None,
            loss_dev='%.2E' % Decimal(self.losses["avg"]["dev"]) if self.losses["avg"]["dev"] is not None else None,
            loss_dev_best='%.2E' % Decimal(self.losses["best"]["dev"]) if self.losses["best"]["dev"] is not None else None,
            plateau=abs(self.last_update - self.epoch) > 3 if self.last_update is not None else None)

    def run(self, epochs=None):
        """
        Run the training with a batch size

        :param epochs: (default infinite - early stop)
        :param batch_size:
        :return:
        """

        # Set the epochs
        self.epochs = epochs

        # Run the epochs
        condition = True
        while condition:
            self.best_model_found = False
            t0 = time.time()

            # Run the trainer normally
            self.model.train()
            self.fit_epoch()

            # Run the trainer normally
            with torch.no_grad():
                self.model.eval()
                self.eval_epoch()

                # Synchronize at the end of each epoch
                self.sync_epoch() if self.sync_epoch is not None else None

                # Checkpoint
                self.checkpoint_epoch() if self.checkpoint_epoch is not None else None

                # Set time elapsed
                self.time_epoch = int(time.time() - t0)

                self.epoch += 1
                if self.epochs is not None:
                    condition = self.epoch < self.epochs


class CTNetTrainer(DefaultTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_kwargs,
                 train_loader,
                 dev_loader,
                 criterion=BCELoss(),
                 epoch=0,
                 epochs=1,
                 model_path='ctnet.pth',
                 checkpoint_path='ctnet.ckpt.pth',
                 continue_from=None,
                 shuffle=True
                 ):
        super(CTNetTrainer, self).__init__(
            model=model,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            criterion=criterion,
            train_loader=train_loader,
            dev_loader=dev_loader,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            fit_epoch=self.fit_epoch,
            eval_epoch=self.eval_epoch,
            checkpoint_epoch=self.checkpoint_epoch,
            sync_epoch=self.sync_epoch)
        self.shuffle = shuffle
        if continue_from is not None:
            try:
                self.model.load_state_dict(torch.load(continue_from))
                self.epoch=epoch
                self.last_update = self.epoch
            except:
                print(">> Could not restart from {continue_from}... ".format(continue_from=continue_from))

    def fit_epoch(self, iter=1):
        # self.train_loader.dataset.shuffle() if self.shuffle else None
        self.model.train()
        for i, (x, y, _, _, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=self.desc()):
            x = x.cuda()
            y = y.cuda()
            self.fit(x=x, y=y)
        self.losses["avg"]["train"] = np.mean(self.losses["history"]["train"][-len(self.train_loader):])
        self.f1_score["avg"]["train"]= np.mean(self.f1_score["history"]["train"][-len(self.train_loader):])

    def fit(self, x, y):
        self.optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        _y = self.model(x)
        loss = self.criterion(_y, y)
        loss = loss / _y.size(1)  # average the loss by minibatch
        loss.backward()

        self.optimizer.step()
        self.losses["history"]["train"].append(loss.item())
        self.f1_score["history"]["train"].append(f1_score(y, _y))



    def eval_epoch(self):
        self.flag = True
        # self.dev_loader.dataset.shuffle() if self.shuffle else None
        for i, (x, y, _, ids, _) in enumerate(self.dev_loader):
            self.eval(x=x, y=y, ids=ids)

        loss_dev = np.mean(self.losses["history"]["dev"][-len(self.dev_loader):])
        self.losses["avg"]["dev"] = loss_dev

        f1_dev = np.mean(self.f1_score["history"]["dev"][-len(self.dev_loader):])
        self.f1_score["avg"]["dev"] = f1_dev

        # if self.f1_score["best"]["dev"] is None or self.f1_score["avg"]["dev"] > self.f1_score["best"]["dev"]:
        if self.losses["best"]["dev"] is None or self.losses["avg"]["dev"] < self.losses["best"]["dev"]:
            self.last_update = self.epoch
            self.best_model_found=True
            self.losses["best"]["dev"]      =   self.losses["avg"]["dev"]
            self.f1_score["best"]["dev"]    =   self.f1_score["avg"]["dev"]

        self.plateau = abs(self.last_update - self.epoch)

    def checkpoint_epoch(self):
        pass

    def sync_epoch(self):
        if self.best_model_found:
            os.makedirs(parent(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)


    def eval(self, x, y, ids=None):
        self.model.eval()
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            _y = self.model(x)
            loss = self.criterion(_y, y)
            loss = loss / _y.size(1)  # average the loss by minibatch
            self.losses["history"]["dev"].append(loss.item())
            self.f1_score["history"]["dev"].append(f1_score(y, _y))
            if self.flag:
                _ycp = np.argwhere(np.array(_y[0].cpu().detach().numpy()) >= 0.5)
                xcp = np.argwhere(np.array(x[0].cpu().detach().numpy()) == 1.0)
                ycp = np.argwhere(np.array(y[0].cpu().detach().numpy()) == 1.0)
                output_dir = f"__data__/eval_xyz/{self.model.id}/{self.epoch}"
                output_dir+= ids[0] if ids is not None else ""
                os.makedirs(output_dir, exist_ok=True)
                np.savetxt("{output_dir}/ai3d_y.xyz".format(output_dir=output_dir), _ycp)
                np.savetxt("{output_dir}/x.xyz".format(output_dir=output_dir), xcp)
                np.savetxt("{output_dir}/y.xyz".format(output_dir=output_dir), ycp)
                self.flag = False

