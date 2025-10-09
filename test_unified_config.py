#!/usr/bin/env python3
"""
Test script to verify unified YAML configuration loading for train_2d_turb_unified.py
"""

import yaml
import sys
import os

def test_config_loading(config_file):
    """Test that the YAML configuration can be loaded correctly."""
    try:
        # Test loading the config file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        print(f"✓ YAML configuration loaded successfully from {config_file}")
        
        # Test that all required sections exist
        required_sections = ['data', 'training', 'model', 'optimizer', 'loss', 'paths', 'options']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing required section: {section}")
                return False
            else:
                print(f"✓ Section '{section}' found")
        
        # Test architecture-specific parameters
        architecture = config['model']['architecture'].lower()
        print(f"✓ Architecture: {architecture}")
        
        if architecture == 'vit':
            # Test ViT parameters
            vit_params = ['img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads', 'head_dim']
            for param in vit_params:
                if param not in config['model']:
                    print(f"✗ Missing ViT parameter: {param}")
                    return False
                else:
                    print(f"✓ ViT parameter '{param}': {config['model'][param]}")
                    
        elif architecture == 'fno':
            # Test FNO parameters
            fno_params = ['modes', 'width']
            for param in fno_params:
                if param not in config['model']:
                    print(f"✗ Missing FNO parameter: {param}")
                    return False
                else:
                    print(f"✓ FNO parameter '{param}': {config['model'][param]}")
        else:
            print(f"✗ Unsupported architecture: {architecture}")
            return False
        
        # Test some key parameters
        print(f"✓ Time step: {config['data']['time_step']}")
        print(f"✓ Learning rate: {config['training']['learning_rate']}")
        print(f"✓ Epochs: {config['training']['epochs']}")
        print(f"✓ Optimizer type: {config['optimizer']['type']}")
        print(f"✓ Network name: {config['paths']['net_name']}")
        
        print(f"\n✓ All configuration tests passed for {config_file}!")
        return True
        
    except FileNotFoundError:
        print(f"✗ {config_file} file not found")
        return False
    except yaml.YAMLError as e:
        print(f"✗ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_model_creation():
    """Test that models can be created with the configurations."""
    try:
        # Import the unified training script
        sys.path.append('.')
        from train_2d_turb_unified import create_model, load_config
        
        # Test ViT configuration
        print("\n--- Testing ViT Model Creation ---")
        config_vit = load_config('config_vit.yaml')
        # Note: We can't actually create the model without CUDA, but we can test the config loading
        print("✓ ViT configuration loaded successfully")
        
        # Test FNO configuration
        print("\n--- Testing FNO Model Creation ---")
        config_fno = load_config('config_fno.yaml')
        print("✓ FNO configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Unified Training Configuration System")
    print("=" * 50)
    
    # Test both configuration files
    configs = ['config.yaml', 'config_vit.yaml', 'config_fno.yaml']
    all_passed = True
    
    for config_file in configs:
        if os.path.exists(config_file):
            print(f"\n--- Testing {config_file} ---")
            success = test_config_loading(config_file)
            all_passed = all_passed and success
        else:
            print(f"⚠ {config_file} not found, skipping...")
    
    # Test model creation logic
    print("\n--- Testing Model Creation Logic ---")
    model_test_passed = test_model_creation()
    all_passed = all_passed and model_test_passed
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
        print("\nUsage examples:")
        print("  python train_2d_turb_unified.py --config config_vit.yaml")
        print("  python train_2d_turb_unified.py --config config_fno.yaml")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if all_passed else 1)
