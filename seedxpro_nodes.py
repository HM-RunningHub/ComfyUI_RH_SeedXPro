import subprocess
import folder_paths
import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch

def ensure_model_downloaded(model_path, repo_id="ByteDance-Seed/Seed-X-PPO-7B"):
    """
    Ensure model is downloaded, auto-download if not exists
    """
    if not os.path.exists(model_path):
        print(f"Model directory does not exist: {model_path}")
        print(f"Downloading model from Hugging Face: {repo_id}")
        try:
            # Create parent directory
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"Model download completed: {model_path}")
        except Exception as e:
            print(f"Model download failed: {e}")
            raise e
    else:
        print(f"Model already exists: {model_path}")

def split_text_into_chunks(text, max_chunk_size=500):
    """
    Split long text into smaller chunks for translation
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    # Try to split by sentences first
    sentences = re.split(r'[.!?。！？]\s*', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chunk_size:
            current_chunk += sentence + ". " if sentence else ""
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". " if sentence else ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If chunks are still too long, split by characters
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split long chunk by characters
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i:i + max_chunk_size])
    
    return final_chunks

def extract_translation_from_output(output, dst_code):
    """
    Extract translation from model output using multiple strategies
    """
    # Strategy 1: Standard pattern with language code
    patterns = [
        f'<{dst_code}>(.*?)(?:<(?!/)|$)',  # Stop at next tag or end
        f'<{dst_code}>(.*)',               # Everything after the tag
        f'{dst_code}>(.*?)(?:<|$)',        # Without opening bracket
        f'<{dst_code}>\s*(.*?)(?:\n\n|$)', # Stop at double newline or end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if result and len(result) > 0:
                return result
    
    # Strategy 2: Find text after the language code marker
    lines = output.split('\n')
    found_marker = False
    result_lines = []
    
    for line in lines:
        if f'<{dst_code}>' in line or f'{dst_code}>' in line:
            found_marker = True
            # Extract text after the marker in the same line
            parts = re.split(f'<{dst_code}>|{dst_code}>', line, 1)
            if len(parts) > 1 and parts[1].strip():
                result_lines.append(parts[1].strip())
            continue
        
        if found_marker:
            # Stop if we hit another language tag
            if re.search(r'<[a-z]{2}>', line, re.IGNORECASE):
                break
            result_lines.append(line)
    
    if result_lines:
        return '\n'.join(result_lines).strip()
    
    return None

def translate_single_chunk(chunk, src, dst, dst_code, model, tokenizer):
    """
    Translate a single chunk of text
    """
    # Simplified prompt format for better results
    message = f"Translate from {src} to {dst}:\n{chunk}\n\nTranslation in {dst} <{dst_code}>:"
    
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    input_length = inputs['input_ids'].shape[1]
    
    # Conservative token calculation
    max_tokens = min(1024, max(150, len(chunk) * 2))
    
    print(f"Translating chunk (length: {len(chunk)}), max_tokens: {max_tokens}")
    
    # Multiple attempts with different parameters
    for attempt in range(2):
        try:
            if attempt == 0:
                # First attempt: greedy decoding
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                # Second attempt: with sampling
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation
            translation = extract_translation_from_output(res, dst_code)
            
            if translation and len(translation.strip()) > 0:
                print(f"Chunk translated successfully (attempt {attempt + 1})")
                return translation
            else:
                print(f"Attempt {attempt + 1} failed to extract translation")
                print(f"Output: {res}")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            continue
    
    # If all attempts failed, return original chunk with a note
    return f"[Translation failed for: {chunk}]"

def translate(**kwargs):
    try:
        prompt = kwargs.get('prompt')
        original_length = len(prompt)
        
        # Only remove truly problematic control characters
        prompt = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)
        
        # Log if any characters were removed
        if len(prompt) != original_length:
            removed_count = original_length - len(prompt)
            print(f"Warning: Removed {removed_count} problematic control character(s) from input")
        
        if not prompt.strip():
            return "Error: Empty input after cleaning"
        
        src = kwargs.get('from')
        dst = kwargs.get('to')
        dst_code = kwargs.get('dst_code')
        model_path = os.path.join(folder_paths.models_dir, 'Seed-X-PPO-7B')

        # Ensure model is downloaded
        ensure_model_downloaded(model_path)

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as model_error:
            error_msg = f"Failed to load model from {model_path}. Error: {model_error}"
            print(error_msg)
            print("The model files may be incomplete or corrupted.")
            print("Please check the model directory and consider deleting it to re-download:")
            print(f"Model path: {model_path}")
            print("You can delete the entire model directory and run again to re-download the model.")
            return f'Model loading failed: {error_msg}. Please delete model directory and re-download.'

        print(f"Starting translation. Input length: {len(prompt)} characters")
        
        # Split text into manageable chunks if necessary
        chunks = split_text_into_chunks(prompt, max_chunk_size=400)
        print(f"Split into {len(chunks)} chunk(s)")
        
        if len(chunks) == 1:
            # Single chunk, translate directly
            result = translate_single_chunk(chunks[0], src, dst, dst_code, model, tokenizer)
        else:
            # Multiple chunks, translate each and combine
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"Translating chunk {i + 1}/{len(chunks)}")
                translation = translate_single_chunk(chunk, src, dst, dst_code, model, tokenizer)
                translated_chunks.append(translation)
            
            result = ' '.join(translated_chunks)
        
        print(f"Translation completed. Final output length: {len(result)} characters")
        return result
        
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        traceback.print_exc()
        return f'Translation failed: {str(e)}'

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