import json
import pandas as pd
from tqdm import tqdm
test_data_json_path = '/home/work/data/hangyul/mimic_cxr_final/medgemma_test_1105_2.json'
save_dir = '/home/work/data/hangyul/mimic_cxr_final/test_신현주.json'
# md_annotation_original = pd.read_excel("/home/work/hangyul_workspace/ROSALIA/md_sheet/merged_eval.xlsx")
# md_annotation = md_annotation_original[md_annotation_original['quality'] == 'not acceptable']
md_absence_annotation = pd.read_excel('/home/work/hangyul_workspace/ROSALIA/md_sheet/absence_merged_신현주.xlsx')
md_presence_annotation = pd.read_excel('/home/work/hangyul_workspace/ROSALIA/md_sheet/presence_merged_신현주.xlsx')

#md_absence_annotation.loc[md_absence_annotation['sample_key']=='s50926698_findings', 'quality'] = 'Not acceptable'

md_absence_annotation = md_absence_annotation[md_absence_annotation['quality'].str.lower() == 'not acceptable']
md_presence_annotation = md_presence_annotation[md_presence_annotation['quality'].str.lower() == 'not acceptable']

with open(test_data_json_path, 'r') as f:
    test_data = json.load(f)

qa_types = ['negative_reported_major_lesion', 'location_without_any_lesion', 'not_reported_major_lesion', 'grounded_major_lesion']

absence_removed_count = 0
# absence annotation 처리
for idx, row in md_absence_annotation.iterrows():
    sample_key = row['sample_key']
    target = row['target']
    location = row['location']
    question = row['question']
    
    org_qa = test_data[sample_key]['section_qa']
    
    for qa_type in ['negative_reported_major_lesion', 'location_without_any_lesion', 'not_reported_major_lesion']:
        if qa_type not in org_qa.keys():
            continue


for idx, row in md_absence_annotation.iterrows():
    sample_key = row['sample_key']
    target = row['target']
    location = row['location']
    question = row['question']
    
    org_qa = test_data[sample_key]['section_qa']
    
    for qa_type in ['negative_reported_major_lesion', 'location_without_any_lesion', 'not_reported_major_lesion']:
        if qa_type not in org_qa.keys():
            continue
        qa_data = org_qa[qa_type]
        
        # 기존 data를 삭제하기 위해 인덱스와 함께 순회
        to_remove_idx = None
        for i, data in enumerate(qa_data):
            if data['target'] == target and data['location'] == location and data['question'][0] == question:
                to_remove_idx = i
                break  # 일치하는 첫 번째 항목만 삭제
            
        # 삭제
        if to_remove_idx is not None:
            del qa_data[to_remove_idx]
            absence_removed_count += 1


    qa_type = 'grounded_major_lesion'
    qa_data = org_qa[qa_type]

    for lesion_key, lesion_data in qa_data.items():
        
        to_remove_idx = None
        for data_idx, data in enumerate(lesion_data):
            if data['type'] == 'grounding_no_location_extension':
                if data['target'] == target and data['location'] == location and data['question'][0] == question:
                    to_remove_idx = data_idx
                    break
        
        if to_remove_idx is not None:
            del lesion_data[to_remove_idx]
            absence_removed_count += 1
            
print(f"absence annotation removed count: {absence_removed_count}")

# presence annotation 처리
presence_removed_count = 0
for idx, row in md_presence_annotation.iterrows():
    sample_key = row['sample_key']
    target = row['target']
    location = row['location']
    question = row['question']
    
    org_qa = test_data[sample_key]['section_qa']
    qa_type = 'grounded_major_lesion'
    qa_data = org_qa[qa_type]
    
    for lesion_key, lesion_data in qa_data.items():
        presence_removed_count += 1
        if target == lesion_data[0]['target']:
            del qa_data[lesion_key]
            break

print(f"presence annotation removed count: {presence_removed_count}")

json.dump(test_data, open(save_dir, 'w'), indent=4)
