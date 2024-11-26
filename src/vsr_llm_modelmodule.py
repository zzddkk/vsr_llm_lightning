import torch
import lightning as L
import torch.nn as nn
from torcheval.metrics import WordErrorRate
from utils import get_cosine_schedule_with_warmup,make_non_pad_mask
class ModelModule(L.LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model ,self.tokenizer = model ,tokenizer
        self.wer = WordErrorRate()
        self.predict = open("./predict.txt","w")

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.trainer.lr}], weight_decay=self.cfg.trainer.weight_decay, betas=(0.9, 0.98))
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.cfg, int(len(self.trainer.datamodule.train_dataloader()) * self.cfg.trainer.epochs / self.cfg.trainer.gradient_accumulation_steps))
        print(f"steps: {int(len(self.trainer.datamodule.train_dataloader()))}")
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def _step(self,batch,batch_idx,mode):
        if mode == "train":
            inputbatch, targetbatch, input_lengths = batch["video"],batch["target"],batch["input_lengths"]
            batch_size = inputbatch.size(0)
            outputs = self.model(inputbatch, targetbatch, input_lengths)
            self.log("loss", outputs.loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("pbar_loss", outputs.loss, prog_bar=True)
            return outputs.loss

        if mode == "test":
            inputbatch, targetbatch, input_lengths = batch["video"],batch["target"],batch["input_lengths"]
            outputs = self.model(inputbatch, targetbatch, input_lengths)
            self.log("test_loss", outputs.loss, on_step=True, on_epoch=True, batch_size=inputbatch.size(0),sync_dist=True)
            self.log("pbar_test_loss", outputs.loss, prog_bar=True)
            outputs = self.model.generate(batch["video"],batch["target"],batch["input_lengths"])
            res = {"outputs":outputs,"target":batch["target"],"filename":batch["file_path"]}
            self.predict.write(f"{res}\n")
            # outputs = self.model(batch["video"],batch["target"],batch["input_lengths"])    
            # return outputs.loss
            return outputs
        
        if mode == "val":
            inputbatch, targetbatch, input_lengths = batch["video"],batch["target"],batch["input_lengths"]
            batch_size = inputbatch.size(0)
            outputs = self.model(inputbatch, targetbatch, input_lengths)
            self.log("val_loss", outputs.loss, on_step=True,on_epoch=True, batch_size=batch_size,sync_dist=True)
            self.log("pbar_val_loss", outputs.loss, prog_bar=True)
            return outputs.loss

    def training_step(self,batch,batch_idx):
        return self._step(batch,batch_idx,"train")

    def test_step(self,batch,batch_idx):
        outputs = self._step(batch,batch_idx,"test")
        self.wer.update(outputs[0].upper(),batch["target"][0])

    def validation_step(self,batch,batch_idx):
        return self._step(batch,batch_idx,"val")

    def on_train_epoch_start(self):
        print(self.trainer)
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()
    
    def on_test_epoch_start(self):
        self.wer.reset()

    def on_test_epoch_end(self):
        wer = self.wer.compute()
        print(f"WER: {wer*100:.2f}%")
        self.log("wer", wer, on_epoch=True,sync_dist=True)
        