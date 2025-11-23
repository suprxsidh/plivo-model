import json
import random
import os

# --- NOISY DATA POOLS (Keep your existing logic) ---
DIGIT_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

def noisy_digits(number_str, prob_spell=0.4, prob_space=0.6):
    res = []
    for char in number_str:
        if char not in DIGIT_MAP:
            res.append(char)
            continue
        val = DIGIT_MAP[char] if random.random() < prob_spell else char
        res.append(val)
    separator = " " if random.random() < prob_space else ""
    return separator.join(res)

def gen_noisy_phone():
    raw = "".join([str(random.randint(0, 9)) for _ in range(10)])
    return noisy_digits(raw), "PHONE"

def gen_noisy_card():
    # Grouped (realistic) vs Stream
    raw = [str(random.randint(0, 9)) for _ in range(16)]
    if random.random() < 0.7:
        groups = [raw[i:i+4] for i in range(0, 16, 4)]
        noisy_groups = ["".join(g) for g in groups]
        return " ".join([noisy_digits(g, 0.2, 0.0) for g in noisy_groups]), "CREDIT_CARD"
    else:
        return noisy_digits("".join(raw)), "CREDIT_CARD"

def gen_noisy_date():
    months = ["january", "february", "march", "april", "may", "june", 
              "july", "august", "september", "october", "november", "december"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    if random.random() < 0.5:
        return f"{random.choice(months)} {random.randint(1, 30)}", "DATE"
    else:
        return f"next {random.choice(days)}", "DATE"

def gen_noisy_email():
    names = ["user", "contact", "support", "admin", "alex", "sam"]
    return f"{random.choice(names)}{random.randint(1,99)} at gmail dot com", "EMAIL"

def gen_city(): return random.choice(["mumbai", "delhi", "chennai", "bangalore", "pune", "london", "paris"]), "CITY"
def gen_person(): return random.choice(["suprasidh", "aryan", "rohit", "priya", "rahul", "john", "alice"]), "PERSON_NAME"
def gen_location(): return random.choice(["downtown", "uptown", "mg road", "airport", "station"]), "LOCATION"

# --- DISJOINT TEMPLATES (THE KEY CHANGE) ---

TRAIN_TEMPLATES = [
    # PHONE (Direct statements)
    "my number is {PHONE}",
    "call me at {PHONE}",
    "contact me on {PHONE}",
    "mobile number {PHONE}",
    "phone is {PHONE}",
    
    # CARD (Explicit declarations)
    "card number is {CREDIT_CARD}",
    "using card {CREDIT_CARD}",
    "charge to {CREDIT_CARD}",
    "pay with {CREDIT_CARD}",
    "billing card {CREDIT_CARD}",
    
    # DATE (Factual)
    "born on {DATE}",
    "due date {DATE}",
    "scheduled for {DATE}",
    
    # EMAIL & CONTEXT
    "email address {EMAIL}",
    "contact {PERSON_NAME} at {EMAIL}",
    "i live in {CITY}",
    "visit {LOCATION}",
    "name is {PERSON_NAME}",
]

DEV_TEST_TEMPLATES = [
    # PHONE (Action oriented - NEW PHRASING)
    "reach out via {PHONE}",
    "dial {PHONE} for support",
    "ring me at {PHONE}",
    "can you call {PHONE}",
    "text {PHONE} immediately",
    
    # CARD (Transaction oriented - NEW PHRASING)
    "debit account {CREDIT_CARD}",
    "refund to {CREDIT_CARD}",
    "verify payment method {CREDIT_CARD}",
    "card details {CREDIT_CARD}",
    "put it on my visa {CREDIT_CARD}",
    
    # DATE (Casual)
    "see you on {DATE}",
    "happening {DATE}",
    "save the date {DATE}",
    
    # EMAIL & CONTEXT
    "send to {EMAIL}",
    "write to {PERSON_NAME}",
    "located near {CITY}",
    "meeting at {LOCATION}",
    "looking for {PERSON_NAME}",
]

def create_example(idx, mode="train"):
    # Select distinct pool based on mode
    pool = TRAIN_TEMPLATES if mode == "train" else DEV_TEST_TEMPLATES
    template = random.choice(pool)
    
    # Add conversational noise
    if random.random() < 0.2:
        words = template.split()
        if len(words) > 1:
            words.insert(1, random.choice(["um", "uh", "please", "actually"]))
            template = " ".join(words)

    text = template
    entities = []
    
    slots = ["{PHONE}", "{CITY}", "{PERSON_NAME}", "{DATE}", "{CREDIT_CARD}", "{EMAIL}", "{LOCATION}"]
    
    while True:
        earliest_slot = None
        earliest_idx = len(text)
        
        for slot in slots:
            idx_found = text.find(slot)
            if idx_found != -1 and idx_found < earliest_idx:
                earliest_idx = idx_found
                earliest_slot = slot
        
        if earliest_slot is None:
            break
            
        val, label = None, None
        
        if earliest_slot == "{PHONE}": val, label = gen_noisy_phone()
        elif earliest_slot == "{CITY}": val, label = gen_city()
        elif earliest_slot == "{PERSON_NAME}": val, label = gen_person()
        elif earliest_slot == "{DATE}": val, label = gen_noisy_date()
        elif earliest_slot == "{CREDIT_CARD}": val, label = gen_noisy_card()
        elif earliest_slot == "{EMAIL}": val, label = gen_noisy_email()
        elif earliest_slot == "{LOCATION}": val, label = gen_location()
        
        prefix = text[:earliest_idx]
        suffix = text[earliest_idx + len(earliest_slot):]
        
        if label:
            entities.append({
                "start": len(prefix),
                "end": len(prefix) + len(val),
                "label": label
            })
        
        text = prefix + val + suffix

    return {
        "id": f"{mode}_{idx}",
        "text": text,
        "entities": entities
    }

def generate_dataset(num, path, mode):
    data = [create_example(i, mode) for i in range(num)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {num} {mode} examples -> {path}")

if __name__ == "__main__":
    # Generate distinct sets
    generate_dataset(1000, "data/train.jsonl", "train")
    generate_dataset(200, "data/dev.jsonl", "dev")
    generate_dataset(100, "data/test.jsonl", "dev") # Test mirrors Dev logic