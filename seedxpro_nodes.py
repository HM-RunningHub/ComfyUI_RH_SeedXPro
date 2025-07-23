import subprocess
import folder_paths
import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch

def ensure_model_downloaded(model_path, repo_id="ByteDance-Seed/Seed-X-PPO-7B"):
    """
    确保模型已下载，如果不存在则自动下载
    """
    if not os.path.exists(model_path):
        print(f"模型目录不存在: {model_path}")
        print(f"正在从 Hugging Face 下载模型: {repo_id}")
        try:
            # 创建父目录
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 下载模型
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"模型下载完成: {model_path}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            raise e
    else:
        print(f"模型已存在: {model_path}")

def translate(**kwargs):
    try:
        prompt = kwargs.get('prompt')
        prompt = re.sub(r'[\x00-\x1f\x7f]', '', prompt)
        src = kwargs.get('from')
        dst = kwargs.get('to')
        dst_code = kwargs.get('dst_code')
        model_path = os.path.join(folder_paths.models_dir, 'Seed-X-PPO-7B')

        # 确保模型已下载
        ensure_model_downloaded(model_path)

        message = f'No CoT. Translate the following {src} sentence into {dst}:\n{prompt} <{dst_code}>'
        #print(message)

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        inputs = tokenizer(message, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50)
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)

        match = re.search(f'<{dst_code}>(.*)', res)
        if match:
            return match.group(1)
        else:
            return 'failed to translate!'
    except Exception as e:
        print(e)
        return 'failed to translate!'

class RH_SeedXPro_Translator:

    language_code_map = {
        "Arabic": "ar",
        "French": "fr",
        "Malay": "ms",
        "Russian": "ru",
        "Czech": "cs",
        "Croatian": "hr",
        "Norwegian Bokmal": "nb",
        "Swedish": "sv",
        "Danish": "da",
        "Hungarian": "hu",
        "Dutch": "nl",
        "Thai": "th",
        "German": "de",
        "Indonesian": "id",
        "Norwegian": "no",
        "Turkish": "tr",
        "English": "en",
        "Italian": "it",
        "Polish": "pl",
        "Ukrainian": "uk",
        "Spanish": "es",
        "Japanese": "ja",
        "Portuguese": "pt",
        "Vietnamese": "vi",
        "Finnish": "fi",
        "Korean": "ko",
        "Romanian": "ro",
        "Chinese": "zh"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "may the force be with you"}),
                "from": (list(cls.language_code_map.keys()), {'default': 'English'}),
                "to": (list(cls.language_code_map.keys()), {'default': 'Chinese'}),
                "seed": ("INT", {"default": 28, "min": 0, "max": 0xffffffffffffffff,}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "translate"

    # OUTPUT_NODE = True

    CATEGORY = "Runninghub/SeedXPro"
    TITLE = "RunningHub SeedXPro Translator"

    def translate(self, **kwargs):
        kwargs['dst_code'] = self.language_code_map[kwargs.get('to')]
        res = translate(**kwargs)
        print(res)
        return (res,)

NODE_CLASS_MAPPINGS = {
    "RunningHub SeedXPro Translator": RH_SeedXPro_Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub SeedXPro Translator": "RunningHub SeedXPro Translator",
} 