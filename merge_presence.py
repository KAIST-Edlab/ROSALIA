import glob, os 
import pandas as pd 
import pdb
import unicodedata

md_sheet_dir = '/home/work/hangyul_workspace/ROSALIA/md_sheet'
sheet_list = glob.glob(os.path.join(md_sheet_dir, '*merged_*.xlsx'))
presence_sheet_list = [x for x in sheet_list if 'absence' not in x and 'all' not in x]
absence_sheet_list = [x for x in sheet_list if 'presence' not in x and 'all' not in x]

def norm(s):
    return unicodedata.normalize('NFC', s)

# Load all sheets
all_dfs = []
for sheet_path in presence_sheet_list:
    print(f"Loading {sheet_path}...")
    df = pd.read_excel(sheet_path)
    
    # Convert quality column: "not acceptable" -> 1, others -> 0
    if 'quality' in df.columns:
        # Normalize and check for "not acceptable"
        df['quality'] = df['quality'].apply(
            lambda x: 1 if norm(str(x).lower()) == norm('not acceptable') else 0
        )
    
    all_dfs.append(df)
    print(f"  Loaded {len(df)} rows")

# Sum only the quality column across all dataframes
if all_dfs:
    # Use the first dataframe as the base (keep other columns as-is)
    summed_df = all_dfs[0].copy()
    
    # Sum only the quality column across all dataframes
    if 'quality' in summed_df.columns:
        # Initialize quality sum with first dataframe's quality values
        quality_sum = summed_df['quality'].copy()
        
        # Sum quality column from all other dataframes
        for df in all_dfs[1:]:
            if 'quality' in df.columns:
                # Align indices and sum
                quality_sum = quality_sum.add(df['quality'], fill_value=0)
        
        # Update the quality column with the sum
        summed_df['quality'] = quality_sum
        
        # Convert non-zero values to 'Not acceptable'
        summed_df['quality'] = summed_df['quality'].apply(
            lambda x: 'Not acceptable' if pd.notna(x) and x != 0 else 0
        )
    
    output_path = os.path.join(md_sheet_dir, 'presence_merged_all.xlsx')
    summed_df.to_excel(output_path, index=False)
    print(f"\nSaved summed dataframe to {output_path}")
    
    print(f"\nSummed dataframe shape: {summed_df.shape}")
    print(f"Summed dataframe columns: {list(summed_df.columns)}")
    
    # Display the summed dataframe
    print("\nSummed DataFrame:")
    print(summed_df.head())


all_dfs = []
for sheet_path in absence_sheet_list:
    print(f"Loading {sheet_path}...")
    df = pd.read_excel(sheet_path)
    all_dfs.append(df)
    print(f"  Loaded {len(df)} rows")

# Concatenate absence dataframes horizontally and save
if all_dfs:
    concatenated_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    print(f"\nSummed dataframe shape: {concatenated_df.shape}")
    output_path_absence = os.path.join(md_sheet_dir, 'absence_merged_all.xlsx')
    concatenated_df.to_excel(output_path_absence, index=False)
    print(f"\nSaved concatenated dataframe to {output_path_absence}")



