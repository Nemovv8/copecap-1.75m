import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import torch
import h5py
from PIL import Image, ImageFile
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from safetensors.torch import load_file

from src_new.utils_round2 import load_data_for_inference, prep_strings, postprocess_preds

ImageFile.LOAD_TRUNCATED_IMAGES = True

CAPTION_LENGTH = 25
EOS_TOKEN = '.'

def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src_new.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src_new.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src_new.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src_new.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)
    AutoModelForCausalLM.register(SmallCapConfig, SmallCap)

def evaluate_model_round2(args, tokenizer, model, eval_df):
    out = []

    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')

    for idx in tqdm(range(len(eval_df))):
        row = eval_df.iloc[idx]
        image_id = row['image_id']
        file_name = row.get('file_name', None)

        decoder_input_ids = prep_strings('', tokenizer, summary=row['summary'], retrieved_caps=row['caps'], k=args.k, is_test=True)
        decoder_input_ids_tensor = torch.tensor([decoder_input_ids]).to(args.device)

        if args.features_path is not None:
            encoder_outputs = BaseModelOutput(
                last_hidden_state=torch.FloatTensor([
                    features[str(image_id)][()]
                ]).to(args.device)
            )
            with torch.no_grad():
                preds = model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids_tensor,
                    **args.generation_kwargs
                )
        else:
            image = Image.open(os.path.join(args.images_dir, file_name)).convert("RGB")
            feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
            pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
            with torch.no_grad():
                preds = model.generate(
                    pixel_values=pixel_values.to(args.device),
                    decoder_input_ids=decoder_input_ids_tensor,
                    **args.generation_kwargs
                )

        pred = tokenizer.decode(preds[0], skip_special_tokens=True)
        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred})

    return out

def load_model(args, checkpoint_path):
    from src_new.vision_encoder_decoder import SmallCap, SmallCapConfig
    config = SmallCapConfig.from_pretrained(checkpoint_path)
    model = SmallCap(config)
    state_dict = load_file(os.path.join(checkpoint_path, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(args.device)
    return model

def main(args):
    register_model_and_config()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.infer_test or args.disable_rag:
        args.features_path = None

    data = load_data_for_inference(args.annotations_path, round1_caps_path=args.captions_path)
    split = 'test' if args.infer_test else 'val'
    eval_df = pd.DataFrame(data[split])

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = EOS_TOKEN

    args.generation_kwargs = {
        'max_new_tokens': CAPTION_LENGTH,
        'no_repeat_ngram_size': 0,
        'length_penalty': 0.0,
        'num_beams': 3,
        'early_stopping': True,
        'eos_token_id': tokenizer.eos_token_id
    }

    checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
    model = load_model(args, checkpoint_path)

    preds = evaluate_model_round2(args, tokenizer, model, eval_df)

    out_path = os.path.join(checkpoint_path, f"{split}_preds.json")
    with open(out_path, 'w') as f:
        json.dump(preds, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="data/images/")
    parser.add_argument("--features_path", type=str, default='features/val.hdf5')
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json")
    parser.add_argument("--captions_path", type=str, default="data/summary_prompt_dataset.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--decoder_name", type=str, default="gpt2")
    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--disable_rag", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--infer_test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)
