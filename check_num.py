import json
import pdb 
qa_file = "/home/work/data/hangyul/mimic_cxr_not_filtered_qa/mimic_cxr_merged.json"

with open(qa_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)['val']

total_question_count = 0
total_answer_count = 0
total_pos_count = 0

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
                    total_question_count += len(qa_item.get("question", []))
                    total_answer_count += len(qa_item.get("answer", []))
                    if qa_item['seg'] is True: total_pos_count += len(qa_item.get("answer", []))
        elif isinstance(qa_list, list):
            for qa_item in qa_list:
                if not isinstance(qa_item, dict):
                    continue
                total_question_count += len(qa_item.get("question", []))
                total_answer_count += len(qa_item.get("answer", []))
                if qa_item['seg'] is True: total_pos_count += len(qa_item.get("answer", []))

print("총 question 개수:", total_question_count)
print("총 answer 개수:", total_answer_count)
print("총 pos sample 개수:", total_pos_count)