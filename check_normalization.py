#!/usr/bin/env python3
"""
Script to check for data leakage in normalization across all example notebooks.
Identifies notebooks where test data statistics might be used to normalize training data.
"""

import json
import glob
import sys

def check_notebook(notebook_path):
    """Check a single notebook for normalization issues."""
    issues = []
    
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
            
        for cell_idx, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') != 'code':
                continue
                
            source = ''.join(cell.get('source', []))
            lines = source.split('\n')
            
            for line_idx, line in enumerate(lines):
                # Check for concatenation of train and test data
                if 'torch.cat' in line and 'train' in line and 'test' in line:
                    # Get context
                    start = max(0, line_idx - 2)
                    end = min(len(lines), line_idx + 10)
                    context = '\n'.join(lines[start:end])
                    
                    # Check if mean/std is computed on concatenated data
                    if '.mean()' in context or '.std()' in context:
                        issues.append({
                            'cell': cell_idx,
                            'line': line_idx,
                            'context': context,
                            'type': 'potential_leakage'
                        })
                
                # Check for computing statistics on combined data
                if ('train' in line and 'test' in line and 
                    ('.mean()' in line or '.std()' in line)):
                    issues.append({
                        'cell': cell_idx,
                        'line': line_idx,
                        'context': line,
                        'type': 'direct_leakage'
                    })
                    
    except Exception as e:
        return None, str(e)
    
    return issues, None

def main():
    """Main function to check all notebooks."""
    print("Checking all example notebooks for normalization issues...")
    print("=" * 80)
    
    all_notebooks = glob.glob('examples/**/*.ipynb', recursive=True)
    problematic_notebooks = []
    
    for notebook_path in sorted(all_notebooks):
        issues, error = check_notebook(notebook_path)
        
        if error:
            print(f"\n❌ Error reading {notebook_path}: {error}")
            continue
            
        if issues:
            problematic_notebooks.append((notebook_path, issues))
            print(f"\n⚠️  {notebook_path}")
            for issue in issues:
                print(f"   Cell {issue['cell']}, Line {issue['line']}: {issue['type']}")
                print(f"   Context:\n{issue['context']}\n")
    
    print("\n" + "=" * 80)
    print(f"\nTotal notebooks checked: {len(all_notebooks)}")
    print(f"Notebooks with potential issues: {len(problematic_notebooks)}")
    
    if problematic_notebooks:
        print("\n⚠️  Manual review required for the notebooks listed above.")
        return 1
    else:
        print("\n✅ No obvious normalization issues found!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
