import os
import pickle

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import transformers
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score


# import roformer

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class SentimentClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = not self.hparams.adv_train
        if self.hparams.model_type == "xlnet":
            self.xlnet: transformers.models.xlnet.XLNetModel = transformers.XLNetModel.from_pretrained(self.hparams.model_name)
            self.hidden_size = self.xlnet.config.d_model
        elif self.hparams.model_type == "roformer":
            self.xlnet: roformer.RoFormerModel = roformer.RoFormerModel.from_pretrained(self.hparams.model_name, max_position_embeddings=1536)
            self.hidden_size = self.xlnet.config.hidden_size
        else:
            self.xlnet = transformers.AutoModel.from_pretrained(self.hparams.model_name)
            self.hidden_size = self.xlnet.config.hidden_size
        if self.hparams.regression:
            self.criterion = nn.MSELoss()
            self.output_dim = 1
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
            self.output_dim = self.hparams.num_classes
        if self.hparams.layer_norm:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=self.hparams.dropout),
                nn.Linear(self.hidden_size, self.output_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=self.hparams.dropout),
                nn.Linear(self.hidden_size, self.output_dim),
            )
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self.attacker = FGM(self) if self.hparams.adv_train else None
        self.class_weights = None

    def ttl(self, t):
        return t.detach().cpu().numpy()

    def logits_to_prediction(self, logits):
        if not self.hparams.regression:
            return torch.argmax(logits, dim=1)
        prediction = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        prediction[logits < -1.5] = 0
        prediction[(logits >= -1.5) & (logits < -0.5)] = 1
        prediction[(logits >= -0.5) & (logits < 0.5)] = 2
        prediction[(logits >= 0.5) & (logits < 1.5)] = 3
        prediction[logits >= 1.5] = 4
        return prediction

    def forward(self, inputs, output_hidden_states=False):
        is_masked = inputs['is_masked'].bool()
        first_mask = inputs.get('first_mask', None)
        inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        backbone_outputs = self.xlnet(**inputs, output_hidden_states=True)
        masked_outputs = backbone_outputs.last_hidden_state[is_masked]
        if self.hparams.pooling_layers > 1:
            for i in range(2, self.hparams.pooling_layers + 1):
                masked_outputs += backbone_outputs.hidden_states[-i][is_masked]
            masked_outputs /= self.hparams.pooling_layers
        logits = self.classifier(masked_outputs)
        if not output_hidden_states:
            return logits
        hidden_states = ((hs := backbone_outputs.hidden_states)[-1] + hs[-2]) / 2
        pooling_output = torch.einsum("bsh,bs,b->bh", hidden_states, first_mask.float(), 1 / first_mask.float().sum(dim=1))
        return logits, pooling_output

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        inputs = batch["inputs"]
        logits = self(inputs).squeeze(-1)
        if self.hparams.regression:
            labels = labels.float() - 2
        if self.hparams.r_drop:
            loss1 = self.criterion(logits, labels)
            logits_extra = self(inputs).squeeze(-1)
            loss2 = self.criterion(logits, labels)
            kl_loss1 = self.kld(torch.log_softmax(logits_extra, dim=-1), torch.softmax(logits, dim=-1))
            kl_loss2 = self.kld(torch.log_softmax(logits, dim=-1), torch.softmax(logits_extra, dim=-1))
            loss = (loss1 + loss2) / 2 + self.hparams.kl_weight * (kl_loss1 + kl_loss2) / 2
        else:
            loss = self.criterion(logits, labels)
        if self.hparams.regression:
            labels = labels.round().long() + 2
        self.log("train_acc", ((prediction := self.logits_to_prediction(logits)) == labels).sum() / labels.size(0), prog_bar=True, on_step=False, on_epoch=True)
        # self.log("train_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        output = {"prediction": self.ttl(prediction), "labels": self.ttl(labels)}
        if self.automatic_optimization:
            return {"loss": loss} | output
        optimizer = self.optimizers(use_pl_optimizer=True)
        lr_scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        loss /= 2
        self.manual_backward(loss)
        self.attacker.attack(epsilon=self.hparams.attack_epsilon)
        adv_logits = self(inputs).squeeze(-1)
        adv_loss = self.criterion(adv_logits, labels) / 2
        self.manual_backward(adv_loss)
        self.attacker.restore()
        self.clip_gradients(optimizer, gradient_clip_val=self.hparams.manual_gradient_clip_val, gradient_clip_algorithm=self.hparams.manual_gradient_clip_algorithm)
        optimizer.step()
        lr_scheduler.step()
        self.log("loss", (loss + adv_loss).item(), prog_bar=True, on_step=True, on_epoch=True)
        return output

    def training_epoch_end(self, outputs):
        predictions = np.concatenate([x["prediction"] for x in outputs])
        labels = np.concatenate([x["labels"] for x in outputs])
        print()
        accuracy = accuracy_score(labels, predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average="macro")
        print(f"Epoch {self.current_epoch} Train | Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}")

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        inputs = batch["inputs"]
        logits = self(inputs).squeeze(-1)
        if self.hparams.regression:
            labels = labels.float() - 2
        loss = self.criterion(logits, labels)
        if self.hparams.regression:
            labels = labels.round().long() + 2
        prediction = self.logits_to_prediction(logits)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        output = {"prediction": self.ttl(prediction), "labels": self.ttl(labels)}
        if self.hparams.mode == "test" and self.hparams.optimize_f1:
            output = output | {"logits": self.ttl(logits)}
        return output

    def validation_epoch_end(self, outputs):
        predictions = np.concatenate([x["prediction"] for x in outputs])
        labels = np.concatenate([x["labels"] for x in outputs])
        # print(predictions, labels)
        # print(predictions.shape, labels.shape)
        print()
        accuracy = accuracy_score(labels, predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average="macro")
        self.log("val_f1", fscore, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=False, on_step=False, on_epoch=True)
        print(f"Epoch {self.current_epoch} Validate | Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}")

        if self.hparams.mode == "test" and self.hparams.optimize_f1:
            # print(outputs[:3])
            logits = np.concatenate([x["logits"] for x in outputs])
            weighted_prediction = lambda logits, weight: np.argmax(np.einsum("bn,n->bn", logits, weight), axis=1)
            f1_loss_func = lambda weight: -f1_score(labels, weighted_prediction(logits, weight), average="macro")
            class_weights = scipy.optimize.minimize(f1_loss_func, np.ones(logits.shape[1]), method="nelder-mead", options={"maxiter": 5 * 1000, "disp": True}).x
            predictions = weighted_prediction(logits, class_weights)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average="macro")
            print(f"Epoch {self.current_epoch} Validate (Optimized) | Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}")
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            print(class_weights)

    def test_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        logits = self(inputs).squeeze(-1)
        if self.hparams.optimize_f1:
            logits = torch.einsum("bn,n->bn", logits, self.class_weights)
        prediction = self.logits_to_prediction(logits)
        return {"prediction": self.ttl(prediction)} | ({"logits": self.ttl(logits)} if self.hparams.is_extra_output else {})

    def test_epoch_end(self, outputs):
        predictions = np.concatenate([x["prediction"] for x in outputs]).tolist()
        pickle.dump(predictions, open(os.path.join(self.hparams.output_path, "prediction.pkl"), "wb"))
        if self.hparams.is_extra_output:
            # pooling_outputs = np.concatenate([x["pooling_output"] for x in outputs], axis=0).tolist()
            logits = np.concatenate([x["logits"] for x in outputs], axis=0).tolist()
            # pickle.dump({"outputs": pooling_outputs, "logits": logits}, open(os.path.join(self.hparams.output_path, "extra_output.pkl"), "wb"))
            pickle.dump({"logits": logits}, open(os.path.join(self.hparams.output_path, "extra_output.pkl"), "wb"))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, eps=self.hparams.eps)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=self.hparams.num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler,'interval': 'step'}}

class SWASupportModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SWASupportModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, inputs):
        is_masked = inputs.pop('is_masked').bool()
        first_mask = inputs.pop("first_mask", None)
        backbone_outputs = self.backbone(**inputs, output_hidden_states=True)
        masked_outputs = backbone_outputs.last_hidden_state[is_masked]
        logits = self.classifier(masked_outputs)
        return logits

class SWASentimentClassifier(SentimentClassifier):
    def __init__(self, **kwargs):
        super(SWASentimentClassifier, self).__init__(**kwargs)
        self.swa_model = None
        if self.hparams.mode == "test":
            self.check_if_swa_ready()

    def check_if_swa_ready(self):
        if self.swa_model is None:
            self.model = SWASupportModel(self.xlnet, self.classifier)
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=self.average_function)

    def average_function(self, ax: torch.Tensor, x: torch.Tensor, num: int) -> torch.Tensor:
        return ax + (x - ax) / (num + 1)

    def on_train_epoch_start(self) -> None:
        self.check_if_swa_ready()

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        inputs = batch["inputs"]
        logits = self.swa_model(inputs)
        loss = self.criterion(logits, labels)
        prediction = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        output = {"prediction": self.ttl(prediction), "labels": self.ttl(labels)}
        if self.hparams.mode == "test" and self.hparams.optimize_f1:
            output = output | {"logits": self.ttl(logits)}
        return output

    def test_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        logits = self.swa_model(inputs)
        if self.hparams.optimize_f1:
            logits = torch.einsum("bn,n->bn", logits, self.class_weights)
        prediction = torch.argmax(logits, dim=1)
        return {"prediction": self.ttl(prediction)} | ({"logits": self.ttl(logits)} if self.hparams.is_extra_output else {})

    def on_validation_epoch_start(self) -> None:
        self.check_if_swa_ready()
        self.swa_model.update_parameters(self.model)
