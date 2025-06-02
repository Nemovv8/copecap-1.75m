import json

# 输入文件路径
summary_path = "experiments/rag_28M_gpt2/merged_preds_all.json"  # 或 train_preds.json / test_preds.json
retrieved_caps_path = "data/retrieved_caps_resnet50x64.json"
output_path = "summary_prompt_dataset2_28m.json"

# 读取 summary 文件
with open(summary_path, "r") as f:
    summaries = json.load(f)

# 读取 retrieved captions 文件
with open(retrieved_caps_path, "r") as f:
    retrieved_caps = json.load(f)

# 构造合并后的结构
combined = {}
for item in summaries:
    image_id = str(item["image_id"])
    caption = item["caption"]
    
    if image_id not in retrieved_caps:
        print(f"[Warning] image_id {image_id} not found in retrieved_caps.")
        continue

    combined[image_id] = {
        "caps": retrieved_caps[image_id],
        "summary": caption
    }

# 保存到输出文件
with open(output_path, "w") as f:
    json.dump(combined, f, indent=4)

print(f"[Done] Combined file saved to: {output_path}")
