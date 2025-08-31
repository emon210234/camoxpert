import yaml
import argparse

def load_config(config_path="configs/default.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(args, config):
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.img_size:
        config['data']['img_size'] = args.img_size
    return config
