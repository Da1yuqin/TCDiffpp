import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
# import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.group_dataset import AIOZDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class TCDiffpp:
    def __init__(
        self,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        required_dancer_num = 3, 
        window_size = 150,
        split_file = None,
        use_ssm=True,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, smpl 
        repr_dim = pos_dim + rot_dim + 4 
        self.repr_dim = repr_dim
        self.use_ssm = use_ssm

        self.required_dancer_num = required_dancer_num
        self.split_file = split_file
        feature_dim = 438 # cond feats dim


        self.horizon = horizon = window_size

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder( 
             nfeats=repr_dim,
             seq_len=horizon,
             latent_dim=512,
             ff_size=1024,
             num_layers=8, 
             num_heads=8,
             dropout=0.1,
             cond_feature_dim=feature_dim, 
             activation=F.gelu,
             required_dancer_num = required_dancer_num,
             use_ssm = self.use_ssm,
         )


        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )
            print(f"loading ckpt from {checkpoint_path}")

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()
    
    def sort_x_axis(self, x, x_index=5):
        """
        Sort dancers in ascending order based on their x-axis coordinate 
        at the final frame.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_dancers, num_frames, num_features).
            x_index (int): Index of the x-axis coordinate in the feature dimension.

        Returns:
            torch.Tensor: Tensor of the same shape as input, with dancers reordered
                        by their x-axis position at the final frame.
        """
        B, D, F, num_features = x.shape

        # Extract x-axis values at the final frame
        init_x = x[:, :, -1, x_index]  # Shape: (B, D)

        # Sort dancers by x-axis (ascending)
        _, x_order = torch.sort(init_x, dim=1, descending=False)  # Shape: (B, D)

        # Reorder dancers using the sort index
        x_sorted = torch.gather(
            x,
            dim=1,
            index=x_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F, num_features)
        )

        return x_sorted

    def calculate_final_x_y_axes_order(self, x):
        """
        x: Tensor of shape (batch_size, num_dancers, num_frames, num_features)
        Assumes feature index 5 is x-axis, index 4 is y-axis.

        Returns:
            final_x_y_axes_order: Tensor of shape (batch_size, num_dancers * 2)
        """
        B, D, F, num_features = x.shape

        # Get final frame positions
        final_x = x[:, :, -1, 5]  # (B, D)
        final_y = x[:, :, -1, 4]

        # Sort final x-axis: get new order of dancer IDs [0,1,2,3,...]
        _, final_x_order = torch.sort(final_x, dim=1, descending=False)
        _, final_y_order = torch.sort(final_y, dim=1, descending=False)

        # Concat
        final_x_y_axes_order = torch.cat([final_x_order, final_y_order], dim=1)  # (B, D*2)

        return final_x_y_axes_order

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt): 
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path) 
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AIOZDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                required_dancer_num = self.required_dancer_num, 
                split_file = self.split_file,
            
            )
            test_dataset = AIOZDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
                required_dancer_num = self.required_dancer_num, 
                split_file = self.split_file,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size//10,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            # wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights" 
            wdir.mkdir(parents=True, exist_ok=True)


        self.accelerator.wait_for_everyone()
        print("Begin Traning")
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            avg_dis_loss = 0
            # train
            self.train()
            for step, (x, cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                # at the init frame to ensure consistent dancer ordering.
                sorted_x = self.sort_x_axis(x)
                sorted_x = sorted_x.to(self.accelerator.device)

                # Compute the dancers' final x- and y-axis order indices.
                # The output encodes the swap pattern between initial dancer IDs 
                # [0, 1, ..., N-1] and their spatial order at the end of the clip.
                final_x_y_axes_order = self.calculate_final_x_y_axes_order(sorted_x)
                final_x_y_axes_order = final_x_y_axes_order.to(self.accelerator.device)

                # loss 
                total_loss, (loss, v_loss, fk_loss, foot_loss, dis_loss) = self.diffusion( 
                    x, cond, t_override=None 
                )

                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    avg_dis_loss += dis_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model, log info, visualization for testing(from val dataset)
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    avg_dis_loss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                        "Dis Loss": avg_dis_loss,
                    }
                    # wandb.log(log_dict)
                    print(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt")) # save ckpt
                    # generate a sample
                    render_count = 2
                    # shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
                    shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))

                    x = x.to(self.accelerator.device) 
                    x_traj_xy = x[:,:,:,[4,4+1]] 
                    bs, dn, seq, c = x_traj_xy.shape
                    x_traj = torch.zeros(bs, dn, seq, 3).to(x_traj_xy)
                    x_traj[:,:,:,[0,1]] = x_traj_xy[:,:,:,[0,1]] 

                    cond = cond.to(self.accelerator.device) 

                    self.diffusion.render_sample( 
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        # fk_out='./runs/fk_out', 
                        name=wavnames[:render_count],
                        sound=True,
                        required_dancer_num = self.required_dancer_num,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        
       
        # if self.accelerator.is_main_process:
        #     wandb.run.finish()

    
    def val_loop(self, opt): 
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path) 
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AIOZDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                required_dancer_num = self.required_dancer_num,
                split_file = self.split_file,
            )
            test_dataset = AIOZDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
                required_dancer_num = self.required_dancer_num, 
                split_file = self.split_file,
            )
            # cache the dataset in case
            pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
            pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch"),
            lambda x: x
        )

        render_count = 30 
        shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
        
        print("Begin Testing")
        self.eval()
        for epoch in range(1, opt.epochs + 1):
            # draw a music from the test dataset
            (x, cond, filename, wavnames) = next(iter(test_data_loader))
            print("Generating Sample")
            x = x.cuda()

            cond = cond.to(x) 

            self.diffusion.render_sample( 
                shape,
                cond[:render_count],
                self.normalizer,
                epoch,
                os.path.join(opt.render_dir, "TRAIN_" + opt.exp_name),
                fk_out = opt.vis_fk_out,  
                name=wavnames[:render_count],
                sound=True,
                required_dancer_num= self.required_dancer_num,
            )
            print(f"[TRAIN-RENDER SAVED at Epoch {epoch}]")

            
            # shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
            shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
            print("Generating Sample")
            # draw a music from the test dataset
            (x, cond, filename, wavnames) = next(iter(test_data_loader))

            x = x.cuda() 
            cond = cond.to(x) 

            self.diffusion.render_sample( 
                shape,
                cond[:render_count],
                self.normalizer,
                epoch,
                os.path.join(opt.render_dir, "TEST_" + opt.exp_name),
                fk_out = opt.vis_fk_out,
                name=wavnames[:render_count],
                sound=True,
                required_dancer_num= self.required_dancer_num,
            )
            print(f"[TEST-RENDER SAVED at Epoch {epoch}]")

            

    def render_sample( 
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, render_len = None,
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0: 
            render_count = len(cond)
        shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label, 
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render,
            required_dancer_num = self.required_dancer_num,
            render_len = render_len,
        )
