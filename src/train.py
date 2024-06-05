from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from tokenizer import tokenizer, tokenize_and_adjust_labels
from model import RobertaForSpanCategorization
#from model import DistilBertForSpanCategorization
from data_preparation import prepare_datasets
import config
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from config import id2label, label2id
import sys
import os
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Append paths for module importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

n_labels = len(id2label)

def divide(a: int, b: int):
    """Utility function for safe division."""
    logging.debug("Dividing {} by {}".format(a, b))
    return a / b if b > 0 else 0

def compute_metrics(p):
    """Calculate and return the performance metrics."""
    logging.info("Computing metrics for predictions")
    predictions, true_labels = p
    predicted_labels = np.where(predictions > 0, 1, 0)
    metrics = {}
    cm = multilabel_confusion_matrix(true_labels.reshape(-1, n_labels), predicted_labels.reshape(-1, n_labels))
    
    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:  # Skip the "O" label
            continue
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{id2label[label_idx]}"] = f1
    
    macro_f1 = np.mean(list(metrics.values()))
    metrics["macro_f1"] = macro_f1
    logging.info("Completed metric computation")
    return metrics

def model_init():
    """Initialize and return the model."""
    logging.info("Initializing model")
    return RobertaForSpanCategorization.from_pretrained("roberta-base", id2label=id2label, label2id=label2id)
    #return DistilBertForSpanCategorization.from_pretrained("DistilBertModel", id2label=id2label, label2id=label2id)

def check_datasets(train_ds, val_ds):
    logging.info(f"Checking dataset sizes: Training - {len(train_ds)}, Validation - {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("One or both datasets are empty. Check the dataset preparation steps.")

def setup_training():
    """Prepare datasets, create the Trainer object, and return it."""
    logging.info("Setting up training configuration")
    train_ds, val_ds = prepare_datasets()
    check_datasets(train_ds, val_ds)

    tokenized_train_ds = train_ds.map(tokenize_and_adjust_labels, remove_columns=train_ds.column_names)
    tokenized_val_ds = val_ds.map(tokenize_and_adjust_labels, remove_columns=val_ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2.5e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        log_level='critical',
        seed=12345
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    logging.info("Training setup complete")
    return trainer

def debug_dataloader(dataloader):
    logging.info("Debugging dataloader")
    try:
        for i, batch in enumerate(dataloader):
            logging.debug(f"Batch {i}: {batch}")
            if i >= 5:  # Just print the first 5 batches
                break
    except Exception as e:
        logging.error(f"Failed to retrieve batch: {e}")

def train():
    logging.info("Starting training process")
    trainer = setup_training()
    batch_count = 0  # Initialize batch count
    try:
        for i, batch in enumerate(trainer.get_train_dataloader()):
            logging.debug(f"Batch {i} - Size: {len(batch)}")
            batch_count += 1
            if i >= 10:
                break
    except Exception as e:
        logging.error(f"Error processing batch {batch_count}: {e}")

    debug_dataloader(trainer.get_train_dataloader())
    trainer.train()
    trainer.model.save_pretrained(config.OUTPUT_DIR)
    logging.info("Training complete and model saved")

if __name__ == "__main__":
    train()
