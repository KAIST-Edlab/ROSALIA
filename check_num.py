import json
import pdb 
qa_file = "/home/work/data/hangyul/mimic_cxr_final/test.json"

with open(qa_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f) # qa_data = json.load(f)['test']

total_question_count = 0
total_answer_count = 0
total_pos_count = 0
opacity_bilat_count = 0

lesion_inference_count = {'edema':0, 'atelectasis':0, 'pneumonia':0, 'opacity':0}
location_count_dict = {'edema':{}, 'atelectasis':{}, 'pneumonia':{}, 'opacity':{}}
change_flag_true_count = 0
change_flag_none_count = 0
new_flag_count = 0

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
                    template_list = qa_item.get("question_type", [])
                    location_txt = str(qa_item.get("grounded_location", "none"))
                    if 'template 4' in template_list: 
                        lesion_inference_count[qa_item['target']] += 1
                        if location_txt in location_count_dict[qa_item['target']].keys(): location_count_dict[qa_item['target']][location_txt] += 1
                        else: location_count_dict[qa_item['target']][location_txt] = 1
                    question_list = qa_item.get("question", [])
                    for question in question_list:
                        if 'Segment the opacity in the left lung and right lung' in question and qa_item['seg'] is True:
                            opacity_bilat_count += 1
                        elif 'Segment the opacity in the right lung and left lung' in question and qa_item['seg'] is True:
                            opacity_bilat_count += 1

                    if qa_item.get("change_flag", False) is True: change_flag_true_count += 1
                    if qa_item.get("change_flag", False) is None: change_flag_none_count += 1
                    if qa_item.get("new_flag", False) is not False: new_flag_count += 1

        elif isinstance(qa_list, list):
            for qa_item in qa_list:
                if not isinstance(qa_item, dict):
                    continue
                total_question_count += len(qa_item.get("question", []))
                total_answer_count += len(qa_item.get("answer", []))
                if qa_item['seg'] is True: total_pos_count += len(qa_item.get("answer", []))
                template_list = qa_item.get("question_type", [])
                location_txt = str(qa_item.get("grounded_location", "none"))
                if 'template 4' in template_list: 
                    lesion_inference_count[qa_item['target']] += 1
                    if location_txt in location_count_dict[qa_item['target']].keys(): location_count_dict[qa_item['target']][location_txt] += 1
                    else: location_count_dict[qa_item['target']][location_txt] = 1
                question_list = qa_item.get("question", [])
                for question in question_list:
                    if 'Segment the opacity in the left lung and right lung' in question and qa_item['seg'] is True:
                        opacity_bilat_count += 1
                    elif 'Segment the opacity in the right lung and left lung' in question and qa_item['seg'] is True:
                        opacity_bilat_count += 1

                if qa_item.get("change_flag", False) is True: change_flag_true_count += 1
                if qa_item.get("change_flag", False) is None: change_flag_none_count += 1
                if qa_item.get("new_flag", False) is not False: new_flag_count += 1

print("총 question 개수:", total_question_count)
print("총 answer 개수:", total_answer_count)
print("총 pos sample 개수:", total_pos_count)
print(f"lesion inference distribution: {lesion_inference_count}")
for k, v in location_count_dict.items():
    print(f"lesion inference location distribution - {k}: {v}")
print(f"opacity bilat count: {opacity_bilat_count}")
print(f'change_flag true : {change_flag_true_count} / change_flag none: {change_flag_none_count} / new_flag: {new_flag_count}')