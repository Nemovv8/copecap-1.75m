import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['WANDB_DISABLED'] = 'true'
import argparse

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from src_new.vision_encoder_decoder import SmallCap, SmallCapConfig
from src_new.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src_new.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src_new.opt import ThisOPTConfig, ThisOPTForCausalLM

from src_new.utils_round2 import *  # ✅ 替换 utils.py
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from transformers.modeling_outputs import BaseModelOutput
import torch


PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    keys = batch[0].keys()
    return {key: [d[key] for d in batch] for key in keys}

def get_model_and_auxiliaries(args):
    if "xglm" in args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    elif "opt" in args.decoder_name:
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
    else:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    model = SmallCap.from_encoder_decoder_pretrained(
        args.encoder_name, args.decoder_name,
        cross_attention_reduce_factor=cross_attention_reduce_factor
    )
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder
    model.config.max_length = CAPTION_LENGTH
    model.config.rag = not args.disable_rag

    for param in model.encoder.parameters():
        param.requires_grad = False

    if "xglm" in args.decoder_name or "opt" in args.decoder_name:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'encoder_attn' not in name:
                    param.requires_grad = False
    else:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Training a model with {num_trainable_params} trainable parameters.")

    return model, tokenizer, feature_extractor

def get_data(tokenizer, max_length, args):
    data = load_data_for_training(
        args.annotations_path,
        # caps_path=args.captions_path,
        round1_caps_path=args.round1_captions_path
    )
    train_df = pd.DataFrame(data['train'])

    train_dataset = TrainDataset(
        df=train_df,
        features_path=os.path.join(args.features_dir, 'train.hdf5'),
        tokenizer=tokenizer,
        rag=not args.disable_rag,
        # template_path=args.template_path,
        k=args.k,
        max_caption_length=max_length
    )

    return train_dataset

# class SamplePredictionCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, **kwargs):
#         trainer = kwargs['trainer']
#         model = trainer.model
#         tokenizer = trainer.tokenizer
#         dataset = trainer.train_dataset
#         device = model.device

#         print("\n📣 [SamplePredictionCallback] Epoch ended, printing 3 samples...")

#         model.eval()
#         with torch.no_grad():
#             for i in range(1, 4):
#                 sample = dataset[-i]
#                 encoder_outputs = sample['encoder_outputs'].unsqueeze(0).to(device)
#                 decoder_input_ids = sample['decoder_input_ids'].unsqueeze(0).to(device)

#                 preds = model.generate(
#                     encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
#                     decoder_input_ids=decoder_input_ids,
#                     max_new_tokens=25,
#                     eos_token_id=tokenizer.eos_token_id
#                 )

#                 print(f"\n[Sample {len(dataset) - i}]")
#                 print("[Prompt (decoder_input_ids)]:", tokenizer.decode(sample['decoder_input_ids']))
#                 print("[Ground Truth (labels)]:", tokenizer.decode([id for id in sample['labels'] if id != -100]))
#                 print("[Model Prediction]:", tokenizer.decode(preds[0], skip_special_tokens=True))

def main(args):
    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    model_type = 'norag' if args.disable_rag else 'rag'
    output_dir = f"{model_type}_{args.attention_size}M_{args.decoder_name}_round2"
    output_dir = os.path.join(args.experiments_dir, output_dir)

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs,
        logging_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )
    # trainer.add_callback(SamplePredictionCallback())
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Second-round Training')
    parser.add_argument("--features_dir", type=str, default="features/")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json")
    parser.add_argument("--experiments_dir", type=str, default="experiments/")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--decoder_name", type=str, default="gpt2")
    parser.add_argument("--attention_size", type=float, default=7)
    parser.add_argument("--train_decoder", action="store_true", default=False)

    parser.add_argument("--disable_rag", action="store_true", default=False)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64")
    # parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json")
    # parser.add_argument("--template_path", type=str, default="src/template_round2.txt")
    parser.add_argument("--round1_captions_path", type=str, default="data/summary_prompt_dataset.json")

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_steps", type=int, default=1)

    args = parser.parse_args()
    main(args)
