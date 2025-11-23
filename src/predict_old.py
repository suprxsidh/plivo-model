import json
import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import re

def is_valid_credit_card(text, start, end):
    """Validate credit card span - should have 13-16 digits"""
    span = text[start:end]
    digit_count = sum(c.isdigit() for c in span)
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    word_count = sum(1 for w in span.lower().split() if w in digit_words)
    # Relaxed from 13 to 12
    return digit_count >= 12 or word_count >= 12

def is_valid_phone(text, start, end):
    """Validate phone number - should have 10+ digits"""
    span = text[start:end]
    digit_count = sum(c.isdigit() for c in span)
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    word_count = sum(1 for w in span.lower().split() if w in digit_words)
    # Relaxed from 10 to 9 for partial matches
    return digit_count >= 9 or word_count >= 9

def is_valid_email(text, start, end):
    """Validate email - should contain @ or 'at' and . or 'dot'"""
    span = text[start:end].lower()
    has_at = "@" in span or " at " in span or span.startswith("at ") or span.endswith(" at")
    has_dot = "." in span or " dot " in span or "dot" in span
    return has_at and has_dot

def is_valid_date(text, start, end):
    """Validate date - should have date indicators"""
    span = text[start:end].lower()
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december",
              "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    
    has_month = any(m in span for m in months)
    has_digit = any(c.isdigit() for c in span)
    
    # STRICTER: Must have BOTH month AND digit, OR have 3+ words with digits
    words = span.split()
    if has_month and has_digit:
        return True
    if has_digit and len(words) >= 3:  # Changed from 2 to 3
        return True
    
    return False  # Reject everything else

def is_valid_person_name(text, start, end):
    """Validate person name - should look like a name"""
    span = text[start:end].strip()
    words = span.split()
    # 1-4 words, each at least 2 chars
    return 1 <= len(words) <= 4 and all(len(w) >= 2 for w in words)

def validate_span(text, start, end, label):
    """Apply validation rules based on entity type"""
    if label == "CREDIT_CARD":
        return is_valid_credit_card(text, start, end)
    elif label == "PHONE":
        return is_valid_phone(text, start, end)
    elif label == "EMAIL":
        return is_valid_email(text, start, end)
    elif label == "DATE":
        return is_valid_date(text, start, end)
    elif label == "PERSON_NAME":
        return is_valid_person_name(text, start, end)
    return True  # CITY and LOCATION don't need strict validation

def bio_to_spans(text, offsets, label_ids):
    """Convert BIO tags to character spans"""
    spans = []
    current_label = None
    current_start = None
    current_end = None
    
    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:  # Special token
            continue
        
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        
        if "-" not in label:
            continue
        
        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
    
    if current_label is not None:
        spans.append((current_start, current_end, current_label))
    
    return spans

def merge_overlapping(spans):
    """Merge overlapping spans, keeping the longest"""
    if not spans:
        return []
    
    sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    merged = [sorted_spans[0]]
    
    for current in sorted_spans[1:]:
        last = merged[-1]
        if current[0] < last[1]:  # Overlap
            if (current[1] - current[0]) > (last[1] - last[0]):
                merged[-1] = current
        else:
            merged.append(current)
    
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    
    results = {}
    
    print(f"Processing {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]
            
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            
            # Convert to spans
            spans = bio_to_spans(text, offsets, pred_ids)
            
            # Validate and filter spans
            validated = [
                (s, e, lab) for s, e, lab in spans 
                if validate_span(text, s, e, lab)
            ]
            
            # Merge overlapping spans
            validated = merge_overlapping(validated)
            
            # Format output
            ents = []
            for s, e, lab in validated:
                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })
            
            results[uid] = ents
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()