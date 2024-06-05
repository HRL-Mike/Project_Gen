import os
import glob
import torch

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoProcessor
from transformers import GPT2Tokenizer


class EndoVis18VQAGPTSentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        
        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)

        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                q_s, an_s = line.split('|')
                q_s = q_s.split('&')
                an_s = an_s.split('&')
                for i in range(len(q_s)):
                    q_a = q_s[i] + '|' + an_s[i]
                    # print(file, q_a)
                    self.vqas.append([file, q_a])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        file_name = self.vqas[idx][0].split('/')[-1]

        # image
        img_loc = os.path.join(seq_path, 'left_frames', file_name.split('_')[0] + '.png')
        image = Image.open(img_loc)
        # prompt
        question, answer = self.vqas[idx][1].split('|')
        
        # 1, 32000, 13, 11889, 29901
        prompt_inputs = self.tokenizer(question, return_tensors='pt')
        additional_ids = [1, 32000]
        additional_ids = torch.tensor(additional_ids).unsqueeze(0)
        input_ids = prompt_inputs['input_ids']
        attention_mask = prompt_inputs['attention_mask']

        pad_length = 40 - input_ids.size(1) - additional_ids.size(1)
        pad_token_id = self.tokenizer.pad_token_id
        
        padded_input_ids = torch.cat([torch.full((1, pad_length), pad_token_id), additional_ids, input_ids], dim=1)
        padded_attention_mask = torch.cat([torch.zeros((1, pad_length), dtype=torch.long), torch.ones((1, additional_ids.size(1)), dtype=torch.long), attention_mask], dim=1)
        prompt_inputs['input_ids'] = padded_input_ids
        prompt_inputs['attention_mask'] = padded_attention_mask
        
        # inputs
        llava_inputs = self.processor(text=question, images=image, return_tensors='pt', padding='max_length', max_length=40, truncation=True)

        return prompt_inputs, llava_inputs, answer
