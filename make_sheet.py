import glob, os 
import pandas as pd 
import pdb
import unicodedata

md_sheet_dir = '/home/work/hangyul_workspace/ROSALIA/md_sheet'
sheet_list = glob.glob(os.path.join(md_sheet_dir, 'raw', '*.xlsx'))
clinician_names = ['윤한결', '신현주', '박현기', '서상훈']

def norm(s):
    return unicodedata.normalize('NFC', s)

for clinician in clinician_names:
    clinician_sheet_list = [x for x in sheet_list if norm(clinician) in norm(x)]
    presence_sheet_list_old = [x for x in clinician_sheet_list if 'new' not in x and 'absence' in x]
    presence_sheet_list_new = [x for x in clinician_sheet_list if 'new' in x and 'absence' in x]

    # Load and merge the first old and new presence sheets
    if presence_sheet_list_old and presence_sheet_list_new:
        df_old = pd.read_excel(presence_sheet_list_old[0])
        df_new = pd.read_excel(presence_sheet_list_new[0])
        
        merged_df = pd.concat([df_old, df_new], ignore_index=True)
        print(f"Merged dataframe shape: {merged_df.shape}")
        print(f"Old dataframe shape: {df_old.shape}")
        print(f"New dataframe shape: {df_new.shape}")
        
        # Save merged dataframe as xlsx
        output_path = os.path.join(md_sheet_dir, f'absence_merged_{norm(clinician)}.xlsx')
        merged_df.to_excel(output_path, index=False)
        print(f"Saved merged dataframe to {output_path}")
    else:
        print("Warning: Could not find both old and new presence sheets")
        merged_df = None

for clinician in clinician_names:
    clinician_sheet_list = [x for x in sheet_list if norm(clinician) in norm(x)]
    presence_sheet_list_old = [x for x in clinician_sheet_list if 'new' not in x and 'presence' in x]
    presence_sheet_list_new = [x for x in clinician_sheet_list if 'new' in x and 'presence' in x]

    # Load and merge the first old and new presence sheets
    if presence_sheet_list_old and presence_sheet_list_new:
        df_old = pd.read_excel(presence_sheet_list_old[0])
        df_new = pd.read_excel(presence_sheet_list_new[0])
        
        # Add numbers to sample names in df_new
        # Find the filename column (could be 'filename' or similar)
        filename_col = None
        for col in df_new.columns:
            if 'filename' in col.lower() or ('sample' in col.lower() and 'name' in col.lower()):
                filename_col = col
                break
        
        if filename_col:
            # Add sequential numbers starting from len(df_old) + 1
            start_num = len(df_old) + 1
            for idx, row_idx in enumerate(df_new.index):
                current_name = str(df_new.loc[row_idx, filename_col])
                current_name_str = current_name.split('_')[0]
                # Add number to the sample name (e.g., "presence_00001" -> "presence_00001_1")
                new_name = f"{current_name_str}_{start_num + idx:05d}"
                df_new.loc[row_idx, filename_col] = new_name
            print(f"Added numbers to {len(df_new)} samples in df_new, starting from {start_num}")
        else:
            print("Warning: Could not find filename column in df_new")
        
        merged_df = pd.concat([df_old, df_new], ignore_index=True)
        print(f"Merged dataframe shape: {merged_df.shape}")
        print(f"Old dataframe shape: {df_old.shape}")
        print(f"New dataframe shape: {df_new.shape}")
        
        # Save merged dataframe as xlsx
        output_path = os.path.join(md_sheet_dir, f'presence_merged_{norm(clinician)}.xlsx')
        merged_df.to_excel(output_path, index=False)
        print(f"Saved merged dataframe to {output_path}")
    else:
        print("Warning: Could not find both old and new presence sheets")
        merged_df = None
