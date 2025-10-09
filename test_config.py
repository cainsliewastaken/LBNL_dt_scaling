#!/usr/bin/env python3
"""
Test script to verify YAML configuration loading for train_2d_turb_ViT.py
"""

import yaml
import sys
import os

def test_config_loading():
    """Test that the YAML configuration can be loaded correctly."""
    try:
        # Test loading the config file
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        print("✓ YAML configuration loaded successfully")
        
        # Test that all required sections exist
        required_sections = ['data', 'training', 'model', 'optimizer', 'loss', 'paths', 'options']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing required section: {section}")
                return False
            else:
                print(f"✓ Section '{section}' found")
        
        # Test some key parameters
        print(f"✓ Time step: {config['data']['time_step']}")
        print(f"✓ Learning rate: {config['training']['learning_rate']}")
        print(f"✓ Epochs: {config['training']['epochs']}")
        print(f"✓ Model depth: {config['model']['depth']}")
        print(f"✓ Optimizer type: {config['optimizer']['type']}")
        
        print("\n✓ All configuration tests passed!")
        return True
        
    except FileNotFoundError:
        print("✗ config.yaml file not found")
        return False
    except yaml.YAMLError as e:
        print(f"✗ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
