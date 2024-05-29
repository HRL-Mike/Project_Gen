import os
import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import GPT2Tokenizer


class EndoVis18VQAGPTSentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail):

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        img = self.clip_processor(images=Image.open(img_loc), return_tensors="pt")

        # question and answer
        question, answer = self.vqas[idx][1].split('|')
        answer = '<|sep|> ' + answer
        question = self.clip_processor(text=question, return_tensors="pt", padding='max_length', max_length=25)
        answer = self.clip_processor(text=answer, return_tensors="pt", padding='max_length', max_length=35)

        return img_loc, img, question, answer


if __name__ == '__main__':
    # data location
    train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    val_seq = [1, 5, 16]
    batch_size = 8

    folder_head = 'D:/1-硕士研究项目/1-数据集/EndoVis-18-VQA/seq_'
    folder_tail = '/vqa/Classification/*.txt'
    model_ver = 'clipgpt2'

    # dataloader
    train_dataset = EndoVis18VQAGPTClassification(train_seq, folder_head, folder_tail, model_ver=model_ver)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = EndoVis18VQAGPTClassification(val_seq, folder_head, folder_tail, model_ver=model_ver)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch in train_dataloader:
        for i, item in enumerate(batch):
            if i == 1:
                print(item['pixel_values'].shape)  # torch.Size([8, 1, 3, 224, 224])
            if i == 2:
                print(item['input_ids'].shape)  # torch.Size([8, 1, 25])
                print(item['attention_mask'].shape)  # torch.Size([8, 1, 25])
        break
