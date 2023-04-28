import time
import os
import argparse
import logging

import torch
import torch.nn as nn

from archs import build_scheduler
from data import create_dataset, create_dataloader
from models import create_model
import utils as util
import utils.option as option


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--opt", required=True, type=str)
    args = parser.parse_args()
    return args


def setup_dataloaer(opt, logger):

    if opt["dist"]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, opt["dist"])
            total_iters = opt["train"]["niter"]
            total_epochs = total_iters // (len(train_loader) - 1) + 1
            if rank == 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), len(train_loader)
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, opt["train"]["niter"]
                    )
                )

        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt["dist"])
            if rank == 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))

    assert train_loader is not None
    assert val_loader is not None

    return train_set, train_loader, val_set, val_loader, total_iters, total_epochs


def main():
    args = parse_args()
    opt = option.parse(opt_path=args.opt, root_path=".", is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if opt["train"].get("resume_state", None) is None:
        util.mkdir_and_rename(
            opt["path"]["experiments_root"]
        )  # rename experiment folder if exists
        util.mkdirs(
            (path for key, path in opt["path"].items() if not key == "experiments_root")
        )
        
        # if os.path.exists('./log'):
        #     if os.name == 'nt':
        #         os.remove("./log")
        #     else:
        #         os.system("rm ./log")
        # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

    # cudnn, seed, logger, device
    torch.backends.cudnn.benchmark = True

    rank = 0

    seed = opt["train"]["manual_seed"]
    if seed is None:
        util.set_random_seed(rank)

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO if rank == 0 else logging.ERROR,
        screen=True,
        tofile=True,
    )

    logger = logging.getLogger("base")

    device= torch.device("cuda:0")

    # create dataset
    (train_set, train_loader, val_dataset, val_loader, total_iters, total_epochs) = setup_dataloaer(opt, logger)

    # create model
    model = create_model(opt)
    model.to(device)

    # create loss
    loss_fn = nn.CrossEntropyLoss().to(device)

    # optimizer
    optim_opt = opt["train"]["optimizer"]
    optim_type = optim_opt.pop("type")
    optim = getattr(torch.optim, optim_type)(params=model.parameters(), **optim_opt)

    # scheduler
    scheduler = build_scheduler(optim, opt["train"]["scheduler"])

    # loading resume state if exists
    start_epoch = 0
    current_step = 0

    # start train process
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    data_time, iter_time = time.time(), time.time()
    avg_data_time = avg_iter_time = 0
    count = 0
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):

            current_step += 1
            count += 1
            if current_step > total_iters:
                break

            data_time = time.time() - data_time
            avg_data_time = (avg_data_time * (count - 1) + data_time) / count

            # feed data
            img, label = train_data
            img = img.to(device)
            label = label.to(device)
            
            # optimize
            optim.zero_grad()

            y_pred = model(img)
            loss = loss_fn(y_pred, label)
            
            loss.backward()
            optim.step()

            # update learning rate
            scheduler.step()
            
            iter_time = time.time() - iter_time
            avg_iter_time = (avg_iter_time * (count - 1) + iter_time) / count
            
            # log
            lr = optim.param_groups[0]["lr"]
            if current_step % opt["logger"]["print_freq"] == 0:
                message = (
                    f"<epoch:{epoch:3d}, iter:{current_step:8,d}, "
                    f"lr:{lr:.3e}> "
                )

                message += f'[time (data): {avg_iter_time:.3f} ({avg_data_time:.3f})] '
                message += "{:s}: {:.4e}; ".format("c_loss", loss.item())
                logger.info(message)

            data_time = time.time()
            iter_time = time.time()
        
            # validate
            if current_step % opt["train"]["val_freq"] == 0:
                acc = validate(val_dataset, val_loader, model, device)
                logger.info("TEST acc:%.5f" % (acc))

            # save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
            
                model_filename = "epoch={}_{}.pth".format(epoch, opt["model"])
                model_path = os.path.join(opt["path"]["models"], model_filename)
                state_dict = model.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                torch.save(state_dict, model_path)

                state = {"epoch": epoch, "iter": current_step, "scheduler": None, "optimizer": optim.state_dict()}
                state_filename = "epoch={}.pth".format(epoch)
                state_path = os.path.join(opt["path"]["training_state"], state_filename)
                torch.save(state, state_path)
    
    # final validate
    acc = validate(val_dataset, val_loader, model, device)
    logger.info("FINAL TEST acc:%.5f" % (acc))

    # save the final model
    logger.info("Saving the final model.")
    model_filename = "latest.pth"
    model_path = os.path.join(opt["path"]["models"], model_filename)
    state_dict = model.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, model_path)


def validate(val_dataset, val_loader, model, device):
    model.eval()
    total_correct = 0
    total_label = 0
    for i, (img, label) in enumerate(val_loader):
        img = img.to(device)

        y_pred = model(img)

        pred = y_pred.detach().cpu().max(1)[1]  # argmax
        total_correct += pred.eq(label.view_as(pred)).sum()
        total_label += len(label)
    acc = total_correct / total_label
    model.train()
    return acc


if __name__=="__main__":
    main()
