from typing import List, Union
import torch
from datasets import tqdm
from torch.utils.data import DataLoader
from transformers import LlamaForSequenceClassification, AutoTokenizer, AdamW

from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.optimizer_args import OptimizerArgs
from src.datasets.text_classification_dataset import TextClassificationDataset
import os

class SequenceClassifierLM:
    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):

        self.env_args = env_args or EnvArgs()
        self.model_args = model_args

        # State variables
        self.model = None
        self.tokenizer = None

    def load(self) -> 'SequenceClassifierLM':
        self.model = LlamaForSequenceClassification.from_pretrained(
            self.model_args.model_name, num_labels=2  # Set num_labels to 2 for binary classification
        ).to(self.env_args.device)

        # Prepare tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name, use_fast=True)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.model_max_length = self.model_args.model_max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        return self

    def fine_tune(self, dataset: TextClassificationDataset, opt_args: OptimizerArgs):
        """ Fine-tune on the text classification dataset """
        opt = AdamW(self.model.parameters(), opt_args.lr, weight_decay=opt_args.weight_decay)
        data_loader: DataLoader = dataset.get_data_loader()

        self._train_one_epoch(data_loader, opt)

        # Save model after training
        self.save_model()

    def _train_one_epoch(
            self,
            train_loader: DataLoader,
            optimizer: AdamW
    ) -> float:
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (AdamW): Optimizer.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        ema_weight = 0.95
        ema_acc, ema_loss = 0.5, 1

        for batch in progress_bar:
            texts: List[str] = batch['text']
            labels: torch.Tensor = batch['source']

            # Tokenization
            encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.model_args.model_max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.env_args.device)
            attention_mask = encoding['attention_mask'].to(self.env_args.device)
            labels = labels.to(self.env_args.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            y_pred = outputs.logits.argmax(1)
            ema_acc = ema_weight * ema_acc + (1 - ema_weight) * (y_pred == labels).float().mean()
            ema_loss = ema_weight * ema_loss + (1 - ema_weight) * loss.item()
            progress_bar.set_postfix({'loss': ema_loss, 'accuracy': ema_acc})

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def _validate(self, val_loader: DataLoader) -> (float, float):
        """
        Evaluates the model on the validation dataset.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            Tuple containing average validation loss and accuracy.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                texts: List[str] = batch['text']
                labels: torch.Tensor = batch['source']

                # Tokenization
                encoding = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.model_args.model_max_length,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.env_args.device)
                attention_mask = encoding['attention_mask'].to(self.env_args.device)
                labels = labels.to(self.env_args.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

                # Calculate accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        return avg_val_loss, accuracy

    @torch.no_grad()
    def classify(self, prompt: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Predicts the class label(s) for the given prompt(s).

        Args:
            prompt (str or List[str]): Input text or list of texts.

        Returns:
            int or List[int]: Predicted class label(s).
        """
        self.model.eval()
        prompts = [prompt] if isinstance(prompt, str) else prompt

        encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.model_args.model_max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.env_args.device)
        attention_mask = encoding['attention_mask'].to(self.env_args.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()

        if isinstance(prompt, str):
            return int(predictions[0])
        return predictions.tolist()

    def save_model(self, save_path: str = "./saved_model"):
        """
        Saves the model and tokenizer after training.

        Args:
            save_path (str): Directory path where the model and tokenizer will be saved.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")
