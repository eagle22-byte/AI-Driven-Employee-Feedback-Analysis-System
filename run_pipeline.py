"""
Complete Pipeline Runner
Runs all steps in sequence: preprocessing -> preparation -> training -> theme extraction
"""

import subprocess
import sys
import os

def run_step(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    
    if not os.path.exists(script_name):
        print(f"[ERROR] {script_name} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        print(f"[SUCCESS] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def main():
    """Run the complete pipeline"""
    print("\n" + "="*60)
    print("AI-DRIVEN EMPLOYEE FEEDBACK ANALYSIS - COMPLETE PIPELINE")
    print("="*60)
    
    steps = [
        ("preprocess_data.py", "Data Preprocessing"),
        ("prepare_dataset.py", "Dataset Preparation with Sentiment Labels"),
        ("train_model.py", "Model Training"),
        ("extract_themes.py", "Theme Extraction")
    ]
    
    for script, description in steps:
        if not run_step(script, description):
            print(f"\n[ERROR] Pipeline stopped at: {description}")
            print("Please fix the error and try again.")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("[SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the model evaluation results above")
    print("2. Check extracted_keywords.txt for themes")
    print("3. Run 'python app.py' to start the web server")
    print("4. Open http://localhost:5000 in your browser")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

