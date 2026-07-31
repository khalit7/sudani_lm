from pathlib import Path
import torch
from tqdm import tqdm
import wandb
from src.factory import Factory
from src.tokenizer.utils import get_tokenizer



class Trainer:
    def __init__(self,config) -> None:

        self.effective_batch_size = config["train"]["effective_batch_size"]
        self.num_epochs           = config["train"]["num_epochs"]
        self.grad_norm_freq       = config["monitor"]["grad_norm"]["freq"]

        # TODO: Improved the tokenizer and loss function usage
        self.tokenizer = get_tokenizer()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.vocab_size = len(self.tokenizer)
        config["model"]["config"]["vocab_size"] = self.vocab_size 

        # get device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"

        factory = Factory(config)
        # get training data
        self.train_dataloader = factory.get_dataloader(config["train"]["dataloader"])
        assert self.train_dataloader is not None
        if self.effective_batch_size%self.train_dataloader.batch_size!=0:
            raise Exception(f"cannot do gradient accumulation with effective batch size  = {self.effective_batch_size} and train dataloader batch size = {self.train_dataloader.batch_size}")
        
        self.train_dataloader_num_steps = len(self.train_dataloader)
        self.grad_acc_every = self.effective_batch_size//self.train_dataloader.batch_size
        self.total_training_steps = self.train_dataloader_num_steps//self.grad_acc_every
        
        # get model,optimiser, and scheduler
        self.model        = factory.get_model()
        self.optimiser    = factory.get_optimiser(self.model.parameters())
        self.lr_scheduler = factory.get_scheduler(self.total_training_steps,self.optimiser)

        self.start_epoch = 0
        self.global_step = 0
        if config["model"]["config"]["is_resume_training"]:
            checkpoint_path = Path("~/sudani_lm/checkpoints").expanduser() / config["model"]["config"]["checkpoint_name"]
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict( checkpoint["model_state_dict"] )
            # .get() so a config that omits load_all_states resumes weights-only instead of
            # dying with a KeyError.
            if config["model"]["config"].get("load_all_states",False):
                self.optimiser.load_state_dict( checkpoint["optimiser_state_dict"] )
                # Key must match what _save_checkpoint writes; it used to read "lr_state_dict"
                # against a checkpoint written as "lr_scheduler_state_dict", so every resume
                # with load_all_states raised KeyError.
                self.lr_scheduler.load_state_dict( checkpoint["lr_scheduler_state_dict"] )
                self.start_epoch = checkpoint.get("epoch",0)
                self.global_step = checkpoint.get("step",0)

        # get evaluators
        self.evals = factory.get_evals(self.model,self.device)


        # add info to config
        config["model"]["stats"] = self.model.get_model_stats()
        config["train"]["dataloader"]["num_samples"] = len(self.train_dataloader.dataset)
        for ev in self.evals:
            if ev.dataloader is not None:
                config["eval"][f"{ev.eval_name}_dataloader_num_samples"] = len(ev.dataloader.dataset)
        # wandb run
        self.wandb_run = wandb.init(
            project=config["run"]["project_name"],
            name=config["run"]["run_name"],
            config=config
        )
        self.wandb_run.define_metric("loss/val_loss",summary="min")

        # setup checkpoint
        self.checkpoints_root = Path("~/sudani_lm/checkpoints").expanduser()/ self.wandb_run.project / self.wandb_run.name
        self.checkpoints_root.mkdir(parents=True,exist_ok=True)



    def train(self):

        self.model.to(self.device)
        self.model.train()

        print("Starting training on device = ",self.device)
        self.run_evals(epoch=self.start_epoch,step=self.global_step)

        total_loss = 0
        epoch = self.start_epoch
        for epoch in range(self.start_epoch,self.num_epochs):
            for acc_steps,(X,Y) in enumerate(tqdm(self.train_dataloader),1):

                # run training step with grad accumulation
                X = {k:v.to(self.device) for k,v in X.items() }
                Y = Y.flatten().to(self.device)
                output = self.model(**X)
                loss = self.loss_fn(output.view(-1,self.vocab_size),Y)
                loss = loss/self.grad_acc_every
                loss.backward()

                total_loss += loss.detach().item()

                # check if we need to update weights
                if acc_steps % self.grad_acc_every == 0 :
                    # Monotonic across epochs. It used to be derived from acc_steps, which
                    # restarts every epoch, so on epoch 2 the wandb step went backwards.
                    self.global_step += 1
                    self.optimiser.step()
                    self.lr_scheduler.step()
                    self.log_grad_norm(step=self.global_step)
                    self.optimiser.zero_grad()
                    self.wandb_run.log(
                                        {
                    "loss/train_loss":total_loss,
                    "learning_rate":self.optimiser.param_groups[0]["lr"]
                                        },
                                        step=self.global_step)
                    total_loss = 0

                    self.run_evals(epoch=epoch,step=self.global_step)

        # save final model
        self._save_checkpoint(epoch=epoch,step=self.global_step,checkpoint_name="final.pt")

    def log_grad_norm(self,step):
        # Was `step % freq != 0 and step > 1000`, which with freq=1 never returned early, so
        # calc_grad_norms ran every step — and it calls .item() per layer, forcing a GPU sync
        # each time.
        if step % self.grad_norm_freq != 0:
            return

        for name,norm in self.model.calc_grad_norms().items():
            self.wandb_run.log( {f"grad_norms/{name}":norm},step=step )

    def run_evals(self,epoch,step,):
        # check if we need to run evals
        for eval in self.evals:
            if step == 0:
                will_run_eval = eval.run_at_0
            else:
                will_run_eval = step % eval.frequency == 0

            if will_run_eval:
                print(f"Running Evaluation: {eval.eval_name}")
                params = {
                    "wandb_run":self.wandb_run,
                    "step": step
                    }
                if eval.eval_name == "generation":
                    params.update({"tokenizer":self.tokenizer})

                checkpoint_name = eval.run_eval(**params)
                if checkpoint_name is not None:
                    self._save_checkpoint(epoch=epoch,step=step,checkpoint_name=checkpoint_name)

    def _save_checkpoint(self,epoch,step,checkpoint_name):
        checkpoint_str = \
        {
        "epoch":epoch,
        "step":step,
        "model_state_dict":self.model.state_dict(),
        "optimiser_state_dict":self.optimiser.state_dict(),
        "lr_scheduler_state_dict":self.lr_scheduler.state_dict(),
        }

        torch.save(checkpoint_str,self.checkpoints_root/checkpoint_name)
