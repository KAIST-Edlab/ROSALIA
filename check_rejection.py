import pandas as pd 
import json 
import glob, os 
from tqdm import tqdm 
import pdb
import unicodedata

test_data_json_path = '/home/work/data/hangyul/mimic_cxr_final/medgemma_test_1105_2.json'
sheet_dir = '/home/work/hangyul_workspace/ROSALIA/md_sheet'
clinician_names = ['윤한결', '신현주', '박현기', '서상훈', 'all']
total_sheet_list = glob.glob(os.path.join(sheet_dir, '*.xlsx'))

with open(test_data_json_path, 'r') as f: test_data = json.load(f)

def norm(s):
    return unicodedata.normalize('NFC', s)

for clinician in clinician_names: 
    absence_removed_count = 0
    presence_removed_count = 0
    absence_total_count = 0
    presence_total_count = 0

    sheet_list = [x for x in total_sheet_list if norm(clinician) in norm(x)]
    for sheet in sheet_list: 
        md_annotation_original = pd.read_excel(sheet)
        md_annotation = md_annotation_original[md_annotation_original['quality'].astype(str).str.lower() == 'not acceptable']
        for idx, row in md_annotation.iterrows():
            sample_key = row['sample_key']
            target = row['target']
            location = row['location']
            question = row['question']
            presence_label = row['presence_label']
            if presence_label != 'absence': continue
            
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
                    absence_removed_count += 1

        for idx, row in md_annotation_original.iterrows():
            sample_key = row['sample_key']
            target = row['target']
            location = row['location']
            question = row['question']
            presence_label = row['presence_label']
            if presence_label != 'absence': continue
            
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
                    absence_total_count += 1


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
                    absence_total_count += 1

    
        for idx, row in md_annotation.iterrows():
            sample_key = row['sample_key']
            target = row['target']
            location = row['location']
            question = row['question']
            presence_label = row['presence_label']
            if presence_label != 'presence': continue
            
            org_qa = test_data[sample_key]['section_qa']
            qa_type = 'grounded_major_lesion'
            qa_data = org_qa[qa_type]
            
            for lesion_key, lesion_data in qa_data.items():
                if target == lesion_data[0]['target']:
                    presence_removed_count += 1
                    break

        for idx, row in md_annotation_original.iterrows():
            sample_key = row['sample_key']
            target = row['target']
            location = row['location']
            question = row['question']
            presence_label = row['presence_label']
            if presence_label != 'presence': continue
            
            org_qa = test_data[sample_key]['section_qa']
            qa_type = 'grounded_major_lesion'
            qa_data = org_qa[qa_type]
            
            for lesion_key, lesion_data in qa_data.items():
                if target == lesion_data[0]['target']:
                    presence_total_count += 1
                    break

    pos_rate = 1-(presence_removed_count/presence_total_count)
    neg_rate = 1-(absence_removed_count/absence_total_count)
    total_rate = 1-((absence_removed_count+presence_removed_count)/(absence_total_count+presence_total_count))

    pos_rate *= 100
    neg_rate *= 100
    total_rate *= 100

    print(f'clinician name: {clinician} - total {total_rate:.3f}%({absence_removed_count+presence_removed_count}/{absence_total_count+presence_total_count})/ presence {pos_rate:.3f}%({presence_removed_count}/{presence_total_count}) / absence {neg_rate:.3f}%({absence_removed_count}/{absence_total_count})')