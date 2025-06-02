from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "


def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None, summary_prompt=None):
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True
        truncation = True

    if summary_prompt is not None:
        prefix = (
            "Use the following hint to describe this image:\n\n"
            f"\"{summary_prompt.strip()}\"\n\n"
            "Now, describe this image:"
            )
    elif retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id]
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))

    if is_test:
        return input_ids
    else:
        return input_ids, label_ids


def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred


class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.max_target_length = max_caption_length

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length
                                     + max_caption_length * k
                                     + len(tokenizer.encode(self.template))
                                     + len(tokenizer.encode('\n\n')) * (k-1)
                                     )
            assert k is not None
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        summary_prompt = self.df[idx].get('summary_prompt', None)

        if self.rag:
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k,
                                                     max_length=self.max_target_length,
                                                     summary_prompt=summary_prompt)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer,
                                                     max_length=self.max_target_length,
                                                     summary_prompt=summary_prompt)

        encoder_outputs = self.features[self.df['cocoid'][idx]][()]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs),
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training(annot_path, caps_path=None, summary_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    if summary_path is not None:
        summary_prompts = json.load(open(summary_path))
    else:
        summary_prompts = {}

    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        cocoid_str = str(item['cocoid'])
        caps = retrieved_caps.get(cocoid_str, []) if caps_path is not None else None
        summary = summary_prompts.get(cocoid_str, None)
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': cocoid_str, 'caps': caps, 'text': ' '.join(sentence['tokens']), 'summary_prompt': summary})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data


def load_data_for_inference(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)

    return data
