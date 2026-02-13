
import json
import os
import glob
import logging

class ExternalDatasetLoader:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Default to benchmark/data_external relative to this file
            self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark/data_external")
        else:
            self.base_dir = base_dir

    def load_sharegpt(self, filename="sharegpt/sharegpt_cleaned.json"):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            logging.warning(f"ShareGPT file not found at {path}")
            return []
        
        logging.info(f"Loading ShareGPT from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert to list of conversation strings
                conversations = []
                for item in data:
                    conv_text = ""
                    for turn in item.get('conversations', []):
                        role = turn.get('from', 'unknown')
                        value = turn.get('value', '')
                        conv_text += f"{role}: {value}\n"
                    if conv_text:
                        conversations.append(conv_text)
                return conversations
        except Exception as e:
            logging.error(f"Error loading ShareGPT: {e}")
            return []

    def load_longbench_pg19(self, filename="longbench/moby_dick.txt"):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            logging.warning(f"PG-19 file not found at {path}")
            return ""
        
        logging.info(f"Loading PG-19 from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error loading PG-19: {e}")
            return ""

    def load_thestack(self, filename="thestack/linux_main.c"):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            logging.warning(f"The Stack file not found at {path}")
            return ""
        
        logging.info(f"Loading The Stack from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error loading The Stack: {e}")
            return ""

if __name__ == "__main__":
    # Test loading
    loader = ExternalDatasetLoader()
    sharegpt = loader.load_sharegpt()
    print(f"Loaded {len(sharegpt)} conversations from ShareGPT.")
    
    pg19 = loader.load_longbench_pg19()
    print(f"Loaded {len(pg19)} characters from PG-19.")
    
    code = loader.load_thestack()
    print(f"Loaded {len(code)} characters from The Stack.")
