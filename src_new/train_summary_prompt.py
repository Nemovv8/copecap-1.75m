import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    CLIPFeatureExtractor,
    CLIPVisionModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from gpt2 import ThisGPT2LMHeadModel, ThisGPT2Config


class SummaryPromptDataset(Dataset):
    def __init__(self, data_path, images_dir, tokenizer, feature_extractor, max_length=512):
        with open(data_path, "r") as f:
            raw_data = json.load(f)

        self.data = list(raw_data.items())  # ✅ 现在是 list of (image_id, entry)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, entry = self.data[idx]

        image_path = os.path.join(self.images_dir, entry["file_name"])
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values[0]

        # 构造 Few-shot prompt
        prompt = "Given the following captions:\n"
        for i, cap in enumerate(entry["caps"]):
            prompt += f"{i+1}. {cap.strip()}\n"
        prompt += "\nSummary:"

        # 拼接 ground-truth summary
        full_text = prompt + " " + entry["summary"]

        # Tokenize 输入
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized.input_ids.squeeze(),
            "attention_mask": tokenized.attention_mask.squeeze(),
            "labels": tokenized.input_ids.squeeze(),
            "pixel_values": pixel_values  # 供 cross-attention 使用
        }



def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # ✅ rename

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values
        # "encoder_hidden_states": encoder_hidden_states  # ✅ match model forward
    }



def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/summary_prompt_dataset.json")
    parser.add_argument("--images_dir", type=str, default="data/images/")
    parser.add_argument("--decoder_name", type=str, default="gpt2-medium")
    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=str, default="checkpoints/summary_model/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = tokenizer.eos_token
    decoder_config = ThisGPT2Config.from_pretrained(args.decoder_name)
    decoder_config.add_cross_attention = True
    model = ThisGPT2LMHeadModel.from_pretrained(args.decoder_name, config=decoder_config)

    model.vision_encoder = CLIPVisionModel.from_pretrained(args.encoder_name)
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    dataset = SummaryPromptDataset(
        data_path=args.data_path,
        images_dir=args.images_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()