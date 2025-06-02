import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    ViTImageProcessor,
    ViTModel,
    T5ForConditionalGeneration,
    VisionEncoderDecoderModel,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer
)

class SummaryPromptDataset(Dataset):
    def __init__(self, data_path, images_dir, tokenizer, feature_extractor, max_length=512):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        self.data = list(raw_data.items())
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
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]

        prompt = "Based on the following captions from similar images:\n"
        for i, cap in enumerate(entry["caps"]):
            prompt += f"{i+1}. {cap.strip()}\n"
        prompt += "\nDescribe this image in one sentence:"

        decoder_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
            )

        targets = self.tokenizer(
            entry["summary"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
            )

        return {
            "pixel_values": pixel_values,
            # "decoder_input_ids": decoder_inputs.input_ids[0],
            # "decoder_attention_mask":decoder_inputs.attention_mask[0],
            "labels": targets.input_ids[0]
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        # "decoder_input_ids": torch.stack([item["decoder_input_ids"] for item in batch]),
        # "decoder_attention_mask": torch.stack([item["decoder_attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        # encoder_outputs = model.encoder(pixel_values=inputs["pixel_values"])
        # decoder_input_ids = model.decoder._shift_right(inputs["labels"])
        # outputs = model.decoder(
        #     input_ids=decoder_input_ids,
        #     encoder_outputs=encoder_outputs,
        #     labels=inputs["labels"]
        # )
        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        if model.training:
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"][:1],
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True
                )
                pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                gt = self.tokenizer.decode(inputs["labels"][0], skip_special_tokens=True)
                print("\n Batch Sample:")
                print(f"🔹 Pred: {pred}")
                print(f"🔸 True: {gt}\n{'-'*40}")
        return (loss, outputs) if return_outputs else loss

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/summary_prompt_dataset.json")
    parser.add_argument("--images_dir", type=str, default="data/images/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/summary_t5")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    decoder = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.decoder.config.is_encoder_decoder = True

    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = decoder.config.vocab_size

    full_dataset = SummaryPromptDataset(args.data_path, args.images_dir, tokenizer, feature_extractor)
    train_idx, eval_idx = train_test_split(list(range(len(full_dataset))), test_size=0.05, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    eval_dataset = Subset(full_dataset, eval_idx)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("✅ Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()

# import os
# import json
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, Subset
# from sklearn.model_selection import train_test_split
# from transformers import (
#     T5Tokenizer,
#     ViTImageProcessor,
#     ViTModel,
#     T5ForConditionalGeneration,
#     VisionEncoderDecoderModel,
#     TrainingArguments,
#     EarlyStoppingCallback,
#     Trainer
# )

# class SummaryPromptDataset(Dataset):
#     def __init__(self, data_path, images_dir, tokenizer, feature_extractor, max_length=512):
#         with open(data_path, "r") as f:
#             raw_data = json.load(f)
#         self.data = list(raw_data.items())
#         self.images_dir = images_dir
#         self.tokenizer = tokenizer
#         self.feature_extractor = feature_extractor
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         _, entry = self.data[idx]
#         image_path = os.path.join(self.images_dir, entry["file_name"])
#         image = Image.open(image_path).convert("RGB")
#         pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]

#         labels = self.tokenizer(entry["summary"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt").input_ids[0]

#         return {
#             "pixel_values": pixel_values,
#             "labels": labels
#         }

# def collate_fn(batch):
#     return {
#         "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
#         "labels": torch.stack([item["labels"] for item in batch])
#     }

# def generate_on_eval(trainer, tokenizer, eval_dataset, num_samples=5):
#     model = trainer.model
#     model.eval()
#     for idx in range(min(num_samples, len(eval_dataset))):
#         batch = eval_dataset[idx]
#         pixel_values = batch["pixel_values"].unsqueeze(0).to(model.device)

#         # build the decoder input prompt
#         _, entry = trainer.eval_dataset.dataset[trainer.eval_dataset.indices[idx]]
#         prompt = "Based on the following captions from similar images:\n"
#         for i, cap in enumerate(entry["caps"]):
#             prompt += f"{i+1}. {cap.strip()}\n"
#         prompt += "\nDescribe this image in one sentence:"
#         decoder_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

#         with torch.no_grad():
#             generated_ids = model.generate(
#                 pixel_values=pixel_values,
#                 decoder_input_ids=decoder_input_ids,
#                 max_new_tokens=64,
#                 num_beams=4,
#                 early_stopping=True
#             )

#         pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         gt = tokenizer.decode(batch["labels"], skip_special_tokens=True)

#         print(f"[{idx}]\n🔹 Pred: {pred}\n🔸 True: {gt}\n{'-'*40}")

# def main():
#     from argparse import ArgumentParser
#     parser = ArgumentParser()
#     parser.add_argument("--data_path", type=str, default="data/summary_prompt_dataset.json")
#     parser.add_argument("--images_dir", type=str, default="data/images/")
#     parser.add_argument("--output_dir", type=str, default="checkpoints/summary_t5")
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--lr", type=float, default=5e-5)
#     parser.add_argument("--epochs", type=int, default=10)
#     args = parser.parse_args()

#     tokenizer = T5Tokenizer.from_pretrained("t5-base")
#     feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
#     encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#     decoder = T5ForConditionalGeneration.from_pretrained("t5-base")
#     model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

#     model.config.decoder_start_token_id = tokenizer.pad_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.eos_token_id = tokenizer.eos_token_id
#     model.config.vocab_size = decoder.config.vocab_size

#     full_dataset = SummaryPromptDataset(args.data_path, args.images_dir, tokenizer, feature_extractor)
#     train_idx, eval_idx = train_test_split(list(range(len(full_dataset))), test_size=0.05, random_state=42)
#     train_dataset = Subset(full_dataset, train_idx)
#     eval_dataset = Subset(full_dataset, eval_idx)

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         weight_decay=0.01,
#         logging_steps=10,
#         save_strategy="epoch",
#         evaluation_strategy="epoch",
#         save_total_limit=2,
#         fp16=torch.cuda.is_available(),
#         remove_unused_columns=False,
#         load_best_model_at_end=True,
#         metric_for_best_model="loss"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         tokenizer=tokenizer,
#         data_collator=collate_fn,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
#     )

#     trainer.train()
#     trainer.save_model(args.output_dir)
#     print("✅ Training complete. Model saved to", args.output_dir)

#     print("\n🔍 Sample predictions on eval set:")
#     generate_on_eval(trainer, tokenizer, eval_dataset)

# if __name__ == "__main__":
#     main()