import os

replacements = {
    'Cross Entropy (Baseline)': 'L_CE',
    'Instance-to-Instance Only': 'L_CE + L_Ins2Ins',
    'Instance-to-Class Only': 'L_CE + L_Ins2Ins + L_Ins2Cls',
    'Full HOPE Loss': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls',
    'HOPE w/o Ins2Ins': 'L_CE + L_Ins2Cls + L_Cls2Cls',
    'HOPE w/o Ins2Cls': 'L_CE + L_Ins2Ins + L_Cls2Cls',
    'Exp Triplet + Ins2Cls (4-Class Setup)': 'L_CE + L_Ins2Ins + L_Triplet + L_Cls2Cls (4-Class)',
    'Full HOPE (4-Class Setup)': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls (4-Class)',
    'Exp Triplet + Ins2Cls': 'L_CE + L_Ins2Ins + L_Triplet + L_Cls2Cls',
    'Standard Triplet Only': 'L_CE + L_Triplet',
    'Hierarchical Triplet Only': 'L_CE + L_Hierarchical_Triplet',
    'Exp Hierarchical Triplet + Ins2Cls': 'L_CE + L_Ins2Ins + L_Hierarchical_Triplet + L_Cls2Cls'
}

files_to_update = [
    '/Users/khoale/Downloads/analysis_report.md',
    '/Users/khoale/Downloads/analysis_report.html'
]

for file_path in files_to_update:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            
        for old_str, new_str in replacements.items():
            content = content.replace(old_str, new_str)
            
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated: {file_path}")
