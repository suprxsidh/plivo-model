import json
import random
import os

# --- NOISY DATA POOLS ---
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
    # Feature: GROUPED Digits (Realistic)
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

# --- UNIFIED TEMPLATES (Safe & High Performance) ---
TEMPLATES = [
    "my number is {PHONE}",
    "call me at {PHONE}",
    "contact {PERSON_NAME} at {EMAIL}",
    "i live in {CITY} near {LOCATION}",
    "born on {DATE}",
    
    # OVERSAMPLED CARDS
    "card number is {CREDIT_CARD}",
    "please charge {CREDIT_CARD} for this",
    "pay with card {CREDIT_CARD} thanks",
    "my card details are {CREDIT_CARD}",
    "billing to {CREDIT_CARD} confirmed",
    
    "please call {PHONE} today",
    "name is {PERSON_NAME}",
    "schedule for {DATE} is confirmed",
    "is {EMAIL} your correct email",
    "visit {LOCATION} in {CITY}",
    "reach out via {PHONE}",
    "send to {EMAIL} immediately",
    "meeting in {CITY} on {DATE}",
    "dial {PHONE} for support",
]

def create_example(idx):
    template = random.choice(TEMPLATES)
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
        
        if earliest_slot is None: break
            
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
            entities.append({"start": len(prefix), "end": len(prefix) + len(val), "label": label})
        
        text = prefix + val + suffix

    return {"id": f"gen_{idx}", "text": text, "entities": entities}

def generate_dataset(num, path):
    data = [create_example(i) for i in range(num)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {num} examples -> {path}")

if __name__ == "__main__":
    # Generate consistent data across the board
    generate_dataset(1000, "data/train.jsonl")
    generate_dataset(200, "data/dev.jsonl")
    generate_dataset(100, "data/test.jsonl")