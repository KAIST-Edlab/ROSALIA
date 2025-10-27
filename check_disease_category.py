import json
import pdb 
qa_file = "/home/work/data/hangyul/mimic_cxr_not_filtered_qa/medgemma_merged_final.json"

with open(qa_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)['train']

disease_set = set()

for section_id, section_data in qa_data.items():
    section_qa = section_data.get("section_qa", {})
    for qa_type, qa_list in section_qa.items():
        # if qa_type != 'not_reported_major_lesion' and qa_type != 'grounded_major_lesion':
        #     pdb.set_trace()
        if isinstance(qa_list, dict):
            # e.g., grounded_major_lesion: dict of lists
            for lesion_key, lesion_qa_list in qa_list.items():
                if not isinstance(lesion_qa_list, list):
                    continue
                for qa_item in lesion_qa_list:
                    if not isinstance(qa_item, dict):
                        continue
                    disease_set.add(qa_item['target'])
        elif isinstance(qa_list, list):
            for qa_item in qa_list:
                if not isinstance(qa_item, dict):
                    continue
                disease_set.add(qa_item['target'])

print("Disease set:", disease_set)