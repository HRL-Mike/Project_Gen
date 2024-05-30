import os
import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import CLIPProcessor
from pathlib import Path
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, CLIPModel


class EndoVis18VQAGPTSentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail):
        # image processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.visual_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # text processor
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        special_token = '<|sep|>'
        self.tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})

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

        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue Manipulation',
                       'Tool Manipulation', 'Cutting', 'Cauterization', 'Suction',
                       'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound Sensing',
                       'left-top', 'right-top', 'left-bottom', 'right-bottom']

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        file_name = self.vqas[idx][0].split('/')[-1]

        # img
        img_loc = os.path.join(seq_path, 'left_frames', file_name.split('_')[0] + '.png')
        img = Image.open(img_loc)
        inputs = self.visual_processor(images=img, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)  # 512

        # question and answer
        question, answer = self.vqas[idx][1].split('|')
        answer = '<|sep|> ' + answer
        question_inputs = self.tokenizer(text=question, return_tensors="pt", padding='max_length', max_length=25)
        answer_inputs = self.tokenizer(text=answer, return_tensors="pt", padding='max_length', max_length=35)
        question_attention_mask = torch.squeeze(question_inputs['attention_mask'], 1)
        answer_attention_mask = torch.squeeze(answer_inputs['attention_mask'], 1)
        question_features = self.clip_model.get_text_features(**question)  # 512
        answer_features = self.clip_model.get_text_features(**answer)  # 512

        return img_loc, image_features, question_features, answer_features, question_attention_mask, answer_attention_mask, answer_inputs
