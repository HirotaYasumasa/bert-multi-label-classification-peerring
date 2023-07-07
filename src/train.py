from datetime import datetime
from pathlib import Path

import torch
from classopt import classopt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

import utils as utils


@classopt(default_long=True)
class Args:
    model_name: str = "cl-tohoku/bert-base-japanese-v2"
    dataset_dir: Path = "./datasets/peerring"

    batch_size: int = 16
    epochs: int = 2
    lr: float = 2e-5
    num_warmup_epochs: int = 2
    max_seq_len: int = 128
    num_cl_divisions: int = 5

    date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    device: str = "cuda:0"
    seed: int = 42

    def __post_init__(self):
        utils.set_seed(self.seed)

        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())
        self.labelnames: list[str] = list(self.label2id.keys())

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("outputs") / model_name / self.date
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> list[dict]:
    return utils.load_jsonl(path).to_dict(orient="records")


def main(args: Args):
    def collate_fn(data_list: list[dict]) -> BatchEncoding:
        text = [d["text"] for d in data_list]

        inputs: BatchEncoding = tokenizer(
            text,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
            max_length=args.max_seq_len
        )

        labels = torch.FloatTensor([d["labels"] for d in data_list])
        return BatchEncoding({**inputs, "labels": labels})


    def create_loader(dataset, batch_size=None, shuffle=False):
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )


    def clone_state_dict() -> dict:
        return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


    @torch.inference_mode()
    def evaluate(dataloader: DataLoader) -> dict[str, float]:
        model.eval()
        loss = 0
        total_samples = 0
        gold_labels, pred_labels = [], []

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            batch_size: int = batch['input_ids'].shape[0]
            loss += out.loss.item() * batch_size
            total_samples += batch_size

            threshold = 0.5
            gold_labels += batch['labels'].tolist()
            pred_labels += (torch.sigmoid(out.logits) > threshold).tolist()
        
        accuracy = accuracy_score(gold_labels, pred_labels)
        accuracies = [accuracy_score(np.array(gold_labels)[:, i], np.array(pred_labels)[:, i]) for i in range(len(args.labels))]

        precision_per_labels, recall_per_labels, f1_per_labels, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average= None,
            zero_division=0
        )

        precision = np.mean(precision_per_labels)
        recall = np.mean(recall_per_labels)
        f1 = np.mean(f1_per_labels)

        return (
            {
                "loss": loss / total_samples,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            {
                "labels_accuracy": accuracies,
                "labels_precision": precision_per_labels,
                "labels_recall": recall_per_labels,
                "labels_f1": f1_per_labels,
            },
        )
    

    @torch.inference_mode()
    def test_evaluate(dataloader: DataLoader) -> dict[str, float]:
        model.eval()
        total_loss = 0
        total_samples = 0
        gold_labels, pred_labels, pred_scores = [], [], []

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            batch_size: int = batch['input_ids'].shape[0]
            total_loss += out.loss.item() * batch_size
            total_samples += batch_size

            threshold = 0.5
            gold_labels += batch['labels'].tolist()
            pred_labels += (torch.sigmoid(out.logits) > threshold).tolist()
            pred_scores += torch.sigmoid(out.logits).tolist()

        gold_labels_np = np.array(gold_labels)
        pred_scores_np = np.array(pred_scores)

        roc_auc = roc_auc_score(gold_labels_np, pred_scores_np)
        auc_per_labels = [roc_auc_score(gold_labels_np[:, i], pred_scores_np[:, i]) for i in range(len(args.labels))]

        accuracy = accuracy_score(gold_labels, pred_labels)
        accuracies = [accuracy_score(np.array(gold_labels)[:, i], np.array(pred_labels)[:, i]) for i in range(len(args.labels))]

        precision_per_labels, recall_per_labels, f1_per_labels, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average=None,
            zero_division=0
        )

        precision = np.mean(precision_per_labels)
        recall = np.mean(recall_per_labels)
        f1 = np.mean(f1_per_labels)

        return (
            {
                "loss": total_loss / total_samples,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            },
            {
                "labels_accuracy": accuracies,
                "labels_precision": precision_per_labels,
                "labels_recall": recall_per_labels,
                "labels_f1": f1_per_labels,
                "labels_auc": auc_per_labels,
            },
            {
                "gold_labels": gold_labels,
                "pred_scores": pred_scores,
            }
        )


    def val_log(metrics: dict, fold: int) -> None:
        utils.log(metrics, args.output_dir/ f"fold{fold + 1}" / f"val-log-fold{fold + 1}.csv")
        tqdm.write(
            f"VALID = " 
            f"epoch: {metrics['epoch']} \t"
            f"loss: {metrics['loss']:2.6f}   \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"precision: {metrics['precision']:.4f} \t"
            f"recall: {metrics['recall']:.4f} \t"
            f"f1: {metrics['f1']:.4f}"
        )


    def train_log(metrics: dict, fold: int) -> None:
        utils.log(metrics, args.output_dir/ f"fold{fold + 1}" / f"train-log-fold{fold + 1}.csv")
        tqdm.write(
            f"TRAIN = " 
            f"epoch: {metrics['epoch']} \t"
            f"loss: {metrics['loss']:2.6f}   \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"precision: {metrics['precision']:.4f} \t"
            f"recall: {metrics['recall']:.4f} \t"
            f"f1: {metrics['f1']:.4f}"
        )

    
    def plot_learning_curve(fold: int):
        sns.set_style('whitegrid')

        train_df = pd.read_csv(f"{args.output_dir}/fold{fold + 1}/train-log-fold{fold + 1}.csv")
        val_df = pd.read_csv(f"{args.output_dir}/fold{fold + 1}/val-log-fold{fold + 1}.csv")

        fig, ax = plt.subplots(figsize=(6,4), dpi=300)

        ax.plot(train_df['epoch'], train_df['f1'], label='Training', color='#005AFF', linewidth=1.0)
        ax.plot(val_df['epoch'], val_df['f1'], label='Validation', color='#FF4B00', linewidth=1.0)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='upper left', prop={'size': 10})
        fig.tight_layout()

        plt.savefig(f"{args.output_dir}/fold{fold + 1}/learning_curve_fold{fold + 1}.png", format='png')


    def plot_ROC_curve(gold_labels: list[int], pred_scores: list[int], labels: list[int], labelnames: list[str], fold: int):
        sns.set_style('whitegrid')

        for idx, _ in enumerate(labels):
            name = labelnames[idx]
            gold_labels = np.array(gold_labels)
            pred_scores = np.array(pred_scores)
            fpr, tpr, _ = roc_curve(gold_labels[:, idx], pred_scores[:, idx])
            roc_auc = auc(fpr, tpr)
    
            fig, ax = plt.subplots(figsize=(6,4), dpi=300)
            
            ax.plot(fpr, tpr, color='#03AF7A',
                    linewidth=1.0, label='ROC curve (AUC = %0.2f)' % roc_auc)
            ax.fill_between(fpr, 0, tpr, color='#50C9B7', alpha=0.4)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(loc='lower right', prop={'size': 10})
            fig.tight_layout()

            plt.savefig(f"{args.output_dir}/fold{fold + 1}/ROC_curve_fold{fold + 1}_{name}.png", format='png')
            plt.close()


    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_seq_len
    )
    
    train_val_dataset: list[dict] = load_dataset(args.dataset_dir / "train_val.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test.jsonl")

    kf = KFold(n_splits=args.num_cl_divisions, shuffle=True, random_state=args.seed)

    test_dataloader: DataLoader = create_loader(test_dataset, shuffle=False)

    for i, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        train_dataset = [train_val_dataset[i] for i in train_idx]
        val_dataset = [train_val_dataset[i] for i in val_idx]

        train_dataloader: DataLoader = create_loader(train_dataset, shuffle=True)
        val_dataloader: DataLoader = create_loader(val_dataset, shuffle=False)

        model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                problem_type="multi_label_classification",
                num_labels=len(args.labels),
            )
            .eval()
            .to(args.device, non_blocking=True)
        )

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(train_dataloader) * args.num_warmup_epochs,
            num_training_steps=len(train_dataloader) * args.epochs
        )

        macro_metrics, _ = evaluate(val_dataloader)
        val_metrics = {"epoch": 0, **macro_metrics}
        best_epoch, best_val_f1 = None, val_metrics["f1"]
        best_state_dict = clone_state_dict()
        val_log(val_metrics, i)

        macro_metrics, _ = evaluate(train_dataloader)
        train_metrics = {"epoch": 0, **macro_metrics}
        train_log(train_metrics, i)

        for epoch in trange(args.epochs, dynamic_ncols=True):
            model.train()

            for batch in tqdm(
                train_dataloader,
                total=len(train_dataloader),
                dynamic_ncols=True,
                leave=False,
            ):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                out: SequenceClassifierOutput = model(**batch)
                loss: torch.FloatTensor = out.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            model.eval()
            macro_metrics, _ = evaluate(val_dataloader)
            val_metrics = {"epoch": f"{epoch + 1}", **macro_metrics}
            val_log(val_metrics, i)

            macro_metrics, _ = evaluate(train_dataloader)
            train_metrics = {"epoch": f"{epoch + 1}", **macro_metrics}
            train_log(train_metrics, i)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_state_dict = clone_state_dict()

        model.load_state_dict(best_state_dict)
        model.eval().to(args.device, non_blocking=True)

        val_macro_metrics, _ = evaluate(val_dataloader)
        val_metrics = {"best-epoch": best_epoch, **val_macro_metrics}
        utils.save_json(val_metrics, args.output_dir / f"fold{i + 1}" / "val-metrics.json")

        test_macro_metrics, test_labels_metrics, prediction_data = test_evaluate(test_dataloader)
        test_metrics = {**test_macro_metrics}
        expanded_label_metrics = [{ "label": label, 
                            "accuracy": test_labels_metrics['labels_accuracy'][i], 
                            "precision": test_labels_metrics['labels_precision'][i], 
                            "recall": test_labels_metrics['labels_recall'][i], 
                            "f1": test_labels_metrics['labels_f1'][i],
                            "auc": test_labels_metrics['labels_auc'][i]
                          } for i, label in enumerate(args.labelnames)]
        combined_metrics = {"test_metrics": test_metrics, "label_metrics": expanded_label_metrics}
        utils.save_json(combined_metrics, args.output_dir / f"fold{i + 1}" / "test-metrics.json")

        plot_learning_curve(i)
        plot_ROC_curve(prediction_data["gold_labels"], prediction_data["pred_scores"], args.labels, args.labelnames, i)

    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
