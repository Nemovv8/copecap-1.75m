import json

def print_value_from_json(file_path, key):
    try:
        # 打开并加载 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 检查键是否存在并打印对应值
        if key in data:
            print(f"Value for key '{key}': {data[key]}")
        else:
            print(f"Key '{key}' not found in the JSON file.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 文件路径
    file_path = "data/retrieved_caps_resnet50x64.json"

    # 用户输入的键
    key = input("Enter the key you want to retrieve: ")

    # 调用函数打印值
    print_value_from_json(file_path, key)
