import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
# from src.vision_encoder_decoder import SmallCap
from src_new.utils_for_restval import load_data_for_inference_2, prep_strings, postprocess_preds
ImageFile.LOAD_TRUNCATED_IMAGES = True

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def evaluate_model(args, feature_extractor, tokenizer, model, eval_df, features, retrieved_caps):
    out = []
    template = open(args.template_path).read().strip() + ' '

    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        caps = eval_df['caps'][idx] if 'caps' in eval_df.columns else retrieved_caps.get(str(image_id), [])

        decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                         k=args.k, is_test=True)

        if features is not None and str(image_id) in features:
            encoder_last_hidden_state = torch.FloatTensor([features[str(image_id)][()]])
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                                      decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                      **args.generation_kwargs)
        else:
            image = Image.open(os.path.join(args.images_dir, file_name)).convert("RGB")
            pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                pred = model.generate(pixel_values=pixel_values.to(args.device),
                                      decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                      **args.generation_kwargs)

        pred_text = tokenizer.decode(pred[0])
        pred_text = postprocess_preds(pred_text, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred_text})

    return out

# def load_model(args, checkpoint_path):
#     config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
#     model = AutoModel.from_pretrained(checkpoint_path)
#     model.config = config
#     model.eval()
#     model.to(args.device)
#     return model

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_split(args, feature_extractor, tokenizer, checkpoint_path, split_name):
    data = load_data_for_inference_2(args.annotations_path, args.captions_path)
    eval_df = pd.DataFrame(data[split_name])
    outfile_name = f'{split_name}_preds.json'

    model = load_model(args, checkpoint_path)

    features = None
    if args.features_root is not None:
        feature_path = os.path.join(args.features_root, f"{split_name}.hdf5")
        if os.path.exists(feature_path):
            features = h5py.File(feature_path, 'r')

    retrieved_caps = json.load(open(args.captions_path)) if args.captions_path and os.path.exists(args.captions_path) else {}

    preds = evaluate_model(args, feature_extractor, tokenizer, model, eval_df, features, retrieved_caps)

    with open(os.path.join(checkpoint_path, outfile_name), 'w') as outfile:
        json.dump(preds, outfile)

# def register_model_and_config():
#     from transformers import AutoModelForCausalLM
#     from src.vision_encoder_decoder import SmallCap, SmallCapConfig
#     from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
#     from src.opt import ThisOPTConfig, ThisOPTForCausalLM
#     from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

#     AutoConfig.register("this_xglm", ThisXGLMConfig)
#     AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
#     AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

#     AutoConfig.register("this_opt", ThisOPTConfig)
#     AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
#     AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

#     AutoConfig.register("this_gpt2", ThisGPT2Config)
#     AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
#     AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

#     AutoConfig.register("smallcap", SmallCapConfig)
#     AutoModel.register(SmallCapConfig, SmallCap)

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

def main(args):
    register_model_and_config()
    # print("[INFO] All args:\n", args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    args.generation_kwargs = {
        'max_new_tokens': CAPTION_LENGTH,
        'no_repeat_ngram_size': 0,
        'length_penalty': 0.,
        'num_beams': 3,
        'early_stopping': True,
        'eos_token_id': tokenizer.eos_token_id
    }

    for split_name in ['train', 'val', 'test', 'restval']:
        print(f"Running inference on {split_name} split...")
        infer_one_split(args, feature_extractor, tokenizer, os.path.join(args.model_path, args.checkpoint_path), split_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on train/val/test with optional features')
    parser.add_argument("--images_dir", type=str, default="data/images/", help="Directory where raw images are stored")
    parser.add_argument("--features_root", type=str, default='features/', help="Root dir containing train/val/test.hdf5")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="Karpathy split annotations")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="Retrieved captions (for RAG prompt)")
    parser.add_argument("--template_path", type=str, default="src_new/template.txt", help="Template prompt path")
    parser.add_argument("--model_path", type=str, default='experiments/rag_1.75M_gpt2/', help="Path to model directory")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoint-88560', help="Checkpoint name")
    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--decoder_name", type=str, default="gpt2")
    parser.add_argument("--k", type=int, default=4, help="Top-k retrieved captions to use in prompt")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)
