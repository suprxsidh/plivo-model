import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset import PIIDataset, collate_batch
from labels import LABELS, PII_LABELS
from model import create_model
from collections import Counter

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def compute_class_weights(train_ds, device):
    """Compute class weights for imbalanced data"""
    label_counts = Counter()
    for item in train_ds.items:
        for label_id in item["labels"]:
            if label_id != -100:
                label_counts[label_id] += 1
    
    total = sum(label_counts.values())
    num_classes = len(LABELS)
    weights = torch.ones(num_classes)
    
    for label_id, count in label_counts.items():
        if count > 0:
            weights[label_id] = total / (num_classes * count)
    
    # Boost PII entity weights for higher precision
    pii_labels_idx = [i for i, label in enumerate(LABELS) 
                      if label.split("-")[-1] in PII_LABELS]
    for idx in pii_labels_idx:
        weights[idx] *= 2.0
    
    return weights.to(device)

def evaluate(model, dev_dl, device, loss_fct):
    """Evaluate on dev set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dev_dl:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(1, num_batches)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print(f"Loading datasets...")
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length)
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    print(f"Train examples: {len(train_ds)}, Dev examples: {len(dev_ds)}")
    
    print(f"Creating model...")
    model = create_model(args.model_name)
    model.to(args.device)
    
    # Compute class weights
    class_weights = compute_class_weights(train_ds, args.device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # Learning rate scheduler
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    print(f"Training for {args.epochs} epochs...")
    best_dev_loss = float('inf')
    model.train()
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = running_loss / len(train_dl)
        dev_loss = evaluate(model, dev_dl, args.device, loss_fct)
        
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, dev_loss={dev_loss:.4f}")
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print(f"Saving best model...")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
    
    print(f"Training complete! Best dev loss: {best_dev_loss:.4f}")

if __name__ == "__main__":
    main()