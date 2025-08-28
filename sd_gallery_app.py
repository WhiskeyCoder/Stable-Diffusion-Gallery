#!/usr/bin/env python3
"""
Stable Diffusion Image Gallery
A Flask web app for managing and browsing SD images with metadata extraction
"""

import os
import json
import hashlib
import shutil
import re
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'gallery_images'
DATABASE_FILE = 'image_database.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_database():
    """Load the flat file database"""
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []


def save_database(data):
    """Save data to flat file database"""
    with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_file_hash(filepath):
    """Generate MD5 hash for file to prevent duplicates"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_png_metadata(image_path):
    """Extract metadata from PNG files, handling multiple SD tools"""
    metadata = {}

    try:
        with Image.open(image_path) as img:
            if hasattr(img, 'text'):
                # Extract all text chunks first
                raw_metadata = dict(img.text)
                
                # Special handling for CivitAI EXIF field 37510 - preserve as bytes if possible
                if '37510' in raw_metadata:
                    # Try to get the raw bytes for this field
                    try:
                        # Access the raw text chunk data
                        if hasattr(img, '_getexif') and img._getexif():
                            exif_data = img._getexif()
                            if 37510 in exif_data:
                                # Get the raw bytes for field 37510
                                raw_metadata['37510'] = exif_data[37510]
                                print(f"DEBUG: Preserved raw bytes for field 37510: {type(raw_metadata['37510'])}")
                    except Exception as e:
                        print(f"DEBUG: Could not preserve raw bytes for 37510: {e}")
                
                metadata.update(raw_metadata)

                # Try different metadata formats
                parsed_data = None

                # Method 1: AUTOMATIC1111 format (parameters field)
                if 'parameters' in raw_metadata:
                    parsed_data = parse_a1111_parameters(raw_metadata['parameters'])

                # Method 2: ComfyUI format (workflow/prompt fields)
                elif 'workflow' in raw_metadata or 'prompt' in raw_metadata:
                    parsed_data = parse_comfyui_metadata(raw_metadata)
                
                # Method 2.5: ComfyUI ASCII format (common in newer versions)
                elif any('ASCII' in str(value) for value in raw_metadata.values()):
                    parsed_data = parse_comfyui_ascii_metadata(raw_metadata)
                
                # Method 2.6: ComfyUI extraMetadata format (embedded JSON)
                elif any('extraMetadata' in str(value) for value in raw_metadata.values()):
                    parsed_data = parse_comfyui_extra_metadata(raw_metadata)

                # Method 3: InvokeAI format
                elif 'invokeai_metadata' in raw_metadata or 'sd-metadata' in raw_metadata:
                    parsed_data = parse_invokeai_metadata(raw_metadata)

                # Method 4: NovelAI format
                elif 'Software' in raw_metadata and 'NovelAI' in raw_metadata.get('Software', ''):
                    parsed_data = parse_novelai_metadata(raw_metadata)

                # Method 5: Generic prompt fields
                elif any(key in raw_metadata for key in ['prompt', 'positive', 'negative']):
                    parsed_data = parse_generic_metadata(raw_metadata)

                # Method 6: Check for CivitAI EXIF metadata (field 37510)
                elif '37510' in raw_metadata:
                    print(f"DEBUG: Found field 37510, type: {type(raw_metadata['37510'])}, value: {raw_metadata['37510'][:100]}...")
                    parsed_data = parse_civitai_exif(raw_metadata['37510'])
                
                # Method 7: Check for JSON in any field
                else:
                    parsed_data = parse_json_metadata(raw_metadata)

                # Merge parsed data
                if parsed_data:
                    metadata.update(parsed_data)

                # Debug: Print all available metadata keys for troubleshooting
                print(f"Available metadata keys: {list(raw_metadata.keys())}")
                
                # Debug: Print first few characters of each metadata value to help identify format
                for key, value in raw_metadata.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"Metadata field '{key}' starts with: {value[:200]}...")
                    elif isinstance(value, str):
                        print(f"Metadata field '{key}': {value}")

                # Fallback: If we have any text data but no positive prompt, store it somewhere visible
                if not metadata.get('positive_prompt') and not metadata.get('negative_prompt'):
                    # Look for any field that might contain text data
                    for key, value in raw_metadata.items():
                        if isinstance(value, str) and len(value) > 50:
                            # If this looks like it might contain prompt data, store it as raw text
                            if any(marker in value.lower() for marker in ['girl', 'boy', 'woman', 'man', 'steps:', 'sampler:', 'seed:']):
                                metadata['raw_metadata_text'] = value[:1000]  # Limit length
                                print(f"DEBUG: Stored raw metadata text as fallback: {value[:200]}...")
                                break

    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")

    # Debug: Show what we're returning
    print(f"DEBUG: extract_png_metadata returning: {list(metadata.keys())}")
    if 'positive_prompt' in metadata:
        print(f"DEBUG: Positive prompt found: {metadata['positive_prompt'][:100]}...")
    if 'raw_metadata_text' in metadata:
        print(f"DEBUG: Raw metadata text found: {metadata['raw_metadata_text'][:100]}...")
    
    return metadata


def parse_a1111_parameters(params_text):
    """Parse AUTOMATIC1111 parameters format"""
    metadata = {}

    try:
        # Split positive and negative prompts
        if 'Negative prompt:' in params_text:
            parts = params_text.split('Negative prompt:', 1)
            metadata['positive_prompt'] = parts[0].strip()

            # Extract negative prompt and other parameters
            neg_and_params = parts[1]
            if any(param in neg_and_params for param in
                   ['Steps:', 'Sampler:', 'CFG scale:', 'Seed:', 'Size:', 'Model:']):
                # Find where the actual parameters start
                param_pattern = r'\b(Steps|Sampler|Schedule type|CFG scale|Seed|Size|Model hash|Model|Version|Denoising strength|Hires upscale|Hires steps|Hires upscaler):'
                param_match = re.search(param_pattern, neg_and_params)

                if param_match:
                    metadata['negative_prompt'] = neg_and_params[:param_match.start()].strip()
                    param_string = neg_and_params[param_match.start():]

                    # Parse individual parameters
                    param_pairs = re.findall(r'(\w+(?:\s+\w+)*?):\s*([^,]+?)(?=,\s*\w+(?:\s+\w+)*?:|$)', param_string)
                    for param_name, param_value in param_pairs:
                        clean_name = param_name.strip().lower().replace(' ', '_')
                        metadata[clean_name] = param_value.strip()
                else:
                    metadata['negative_prompt'] = neg_and_params.strip()
            else:
                metadata['negative_prompt'] = neg_and_params.strip()
        else:
            metadata['positive_prompt'] = params_text

    except Exception as e:
        print(f"Error parsing A1111 parameters: {e}")

    return metadata


def parse_comfyui_metadata(raw_metadata):
    """Parse ComfyUI workflow/prompt format"""
    metadata = {}

    try:
        # ComfyUI stores data as JSON in workflow or prompt fields
        import json

        if 'workflow' in raw_metadata:
            try:
                workflow_data = json.loads(raw_metadata['workflow'])
                metadata.update(extract_comfyui_workflow_data(workflow_data))
            except json.JSONDecodeError:
                pass

        if 'prompt' in raw_metadata:
            try:
                prompt_data = json.loads(raw_metadata['prompt'])
                metadata.update(extract_comfyui_prompt_data(prompt_data))
            except json.JSONDecodeError:
                # Sometimes prompt is just text
                metadata['positive_prompt'] = raw_metadata['prompt']

    except Exception as e:
        print(f"Error parsing ComfyUI metadata: {e}")

    return metadata


def extract_comfyui_workflow_data(workflow_data):
    """Extract data from ComfyUI workflow JSON"""
    metadata = {}

    try:
        nodes = workflow_data.get('nodes', [])

        for node in nodes:
            node_type = node.get('type', '')

            # Look for prompt nodes
            if 'prompt' in node_type.lower() or 'text' in node_type.lower():
                widgets = node.get('widgets_values', [])
                if widgets:
                    text = str(widgets[0])
                    if 'negative' in node.get('title', '').lower():
                        metadata['negative_prompt'] = text
                    else:
                        metadata['positive_prompt'] = text

            # Look for sampler nodes
            elif 'sampler' in node_type.lower():
                widgets = node.get('widgets_values', [])
                if len(widgets) >= 3:
                    metadata['steps'] = widgets[0] if isinstance(widgets[0], (int, float)) else None
                    metadata['cfg_scale'] = widgets[1] if isinstance(widgets[1], (int, float)) else None
                    metadata['sampler'] = widgets[2] if isinstance(widgets[2], str) else None

    except Exception as e:
        print(f"Error extracting ComfyUI workflow data: {e}")

    return metadata


def extract_comfyui_prompt_data(prompt_data):
    """Extract data from ComfyUI prompt JSON"""
    metadata = {}

    try:
        # ComfyUI prompt format varies, try to find common patterns
        for key, value in prompt_data.items():
            if isinstance(value, dict):
                inputs = value.get('inputs', {})
                class_type = value.get('class_type', '')

                if 'text' in inputs:
                    if 'negative' in key.lower():
                        metadata['negative_prompt'] = inputs['text']
                    else:
                        metadata['positive_prompt'] = inputs['text']

                if 'KSampler' in class_type:
                    metadata['seed'] = inputs.get('seed')
                    metadata['steps'] = inputs.get('steps')
                    metadata['cfg_scale'] = inputs.get('cfg')
                    metadata['sampler'] = inputs.get('sampler_name')

    except Exception as e:
        print(f"Error extracting ComfyUI prompt data: {e}")

    return metadata


def parse_comfyui_ascii_metadata(raw_metadata):
    """Parse ComfyUI ASCII-encoded metadata format"""
    metadata = {}

    try:
        import json
        
        print("DEBUG: parse_comfyui_ascii_metadata called")
        
        # Look for ASCII field containing JSON data
        for key, value in raw_metadata.items():
            if isinstance(value, str) and value.startswith('ASCII'):
                print(f"DEBUG: Found ASCII field '{key}' with value starting: {value[:100]}...")
                try:
                    # Extract the JSON part after "ASCII"
                    json_start = value.find('{')
                    if json_start != -1:
                        json_str = value[json_start:]
                        print(f"DEBUG: Extracted JSON string starting: {json_str[:100]}...")
                        ascii_data = json.loads(json_str)
                        
                        # Look for extraMetadata field which contains the actual prompt data
                        if 'extraMetadata' in ascii_data:
                            extra_meta_str = ascii_data['extraMetadata']
                            print(f"DEBUG: Found extraMetadata: {extra_meta_str[:100]}...")
                            # The extraMetadata is escaped JSON, so we need to parse it again
                            extra_meta = json.loads(extra_meta_str)
                            
                            # Extract the key information
                            if 'prompt' in extra_meta:
                                metadata['positive_prompt'] = extra_meta['prompt']
                                print(f"DEBUG: Extracted positive prompt: {extra_meta['prompt'][:100]}...")
                            if 'negativePrompt' in extra_meta:
                                metadata['negative_prompt'] = extra_meta['negativePrompt']
                            if 'cfgScale' in extra_meta:
                                metadata['cfg_scale'] = extra_meta['cfgScale']
                            if 'sampler' in extra_meta:
                                metadata['sampler'] = extra_meta['sampler']
                            if 'clipSkip' in extra_meta:
                                metadata['clip_skip'] = extra_meta['clipSkip']
                            if 'steps' in extra_meta:
                                metadata['steps'] = extra_meta['steps']
                            if 'seed' in extra_meta:
                                metadata['seed'] = extra_meta['seed']
                            if 'width' in extra_meta:
                                metadata['width'] = extra_meta['width']
                            if 'height' in extra_meta:
                                metadata['height'] = extra_meta['height']
                            if 'baseModel' in extra_meta:
                                metadata['model_name'] = extra_meta['baseModel']
                            
                            # Extract size as combined field
                            if 'width' in extra_meta and 'height' in extra_meta:
                                metadata['size'] = f"{extra_meta['width']}x{extra_meta['height']}"
                            
                            print(f"DEBUG: Successfully extracted metadata: {metadata}")
                            break
                            
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing ComfyUI ASCII metadata: {e}")
                    continue

    except Exception as e:
        print(f"Error in parse_comfyui_ascii_metadata: {e}")

    return metadata


def parse_comfyui_extra_metadata(raw_metadata):
    """Parse ComfyUI extraMetadata format (embedded JSON)"""
    metadata = {}

    try:
        import json
        
        print("DEBUG: parse_comfyui_extra_metadata called")
        
        # Look for any field containing extraMetadata
        for key, value in raw_metadata.items():
            if isinstance(value, str) and 'extraMetadata' in value:
                print(f"DEBUG: Found field '{key}' containing extraMetadata")
                try:
                    # Try to find the extraMetadata JSON within the string
                    start_marker = '"extraMetadata":"'
                    end_marker = '"}'
                    
                    start_pos = value.find(start_marker)
                    if start_pos != -1:
                        start_pos += len(start_marker)
                        end_pos = value.find(end_marker, start_pos)
                        
                        if end_pos != -1:
                            extra_meta_str = value[start_pos:end_pos]
                            print(f"DEBUG: Extracted extraMetadata string: {extra_meta_str[:200]}...")
                            # The string contains escaped quotes, so we need to unescape it
                            extra_meta_str = extra_meta_str.replace('\\"', '"').replace('\\n', '\n')
                            
                            try:
                                extra_meta = json.loads(extra_meta_str)
                                print(f"DEBUG: Successfully parsed extraMetadata JSON")
                                
                                # Extract the key information
                                if 'prompt' in extra_meta:
                                    metadata['positive_prompt'] = extra_meta['prompt']
                                    print(f"DEBUG: Extracted positive prompt: {extra_meta['prompt'][:100]}...")
                                if 'negativePrompt' in extra_meta:
                                    metadata['negative_prompt'] = extra_meta['negativePrompt']
                                if 'cfgScale' in extra_meta:
                                    metadata['cfg_scale'] = extra_meta['cfgScale']
                                if 'sampler' in extra_meta:
                                    metadata['sampler'] = extra_meta['sampler']
                                if 'clipSkip' in extra_meta:
                                    metadata['clip_skip'] = extra_meta['clipSkip']
                                if 'steps' in extra_meta:
                                    metadata['steps'] = extra_meta['steps']
                                if 'seed' in extra_meta:
                                    metadata['seed'] = extra_meta['seed']
                                if 'width' in extra_meta:
                                    metadata['width'] = extra_meta['width']
                                if 'height' in extra_meta:
                                    metadata['height'] = extra_meta['height']
                                if 'baseModel' in extra_meta:
                                    metadata['model_name'] = extra_meta['baseModel']
                                
                                # Extract size as combined field
                                if 'width' in extra_meta and 'height' in extra_meta:
                                    metadata['size'] = f"{extra_meta['width']}x{extra_meta['height']}"
                                
                                print(f"DEBUG: Successfully extracted metadata: {metadata}")
                                break
                                
                            except json.JSONDecodeError:
                                print(f"DEBUG: JSON parsing failed, trying fallback text extraction")
                                # If JSON parsing fails, try to extract just the prompt text
                                if 'prompt' in extra_meta_str:
                                    # Find the prompt value in the string
                                    prompt_start = extra_meta_str.find('"prompt":"') + 9
                                    prompt_end = extra_meta_str.find('","', prompt_start)
                                    if prompt_end == -1:
                                        prompt_end = extra_meta_str.find('"}', prompt_start)
                                    
                                    if prompt_start > 8 and prompt_end > prompt_start:
                                        metadata['positive_prompt'] = extra_meta_str[prompt_start:prompt_end]
                                        print(f"DEBUG: Fallback extracted prompt: {metadata['positive_prompt'][:100]}...")
                                
                except Exception as e:
                    print(f"Error parsing ComfyUI extraMetadata: {e}")
                    continue

    except Exception as e:
        print(f"Error in parse_comfyui_extra_metadata: {e}")

    return metadata


def parse_invokeai_metadata(raw_metadata):
    """Parse InvokeAI metadata format"""
    metadata = {}

    try:
        import json

        # InvokeAI can store metadata in different fields
        for field in ['invokeai_metadata', 'sd-metadata', 'dream']:
            if field in raw_metadata:
                try:
                    data = json.loads(raw_metadata[field])

                    # Extract common fields
                    metadata['positive_prompt'] = data.get('prompt', '')
                    metadata['negative_prompt'] = data.get('negative_prompt', '')
                    metadata['steps'] = data.get('steps')
                    metadata['sampler'] = data.get('sampler')
                    metadata['cfg_scale'] = data.get('cfg_scale')
                    metadata['seed'] = data.get('seed')
                    metadata['width'] = data.get('width')
                    metadata['height'] = data.get('height')

                    if metadata['width'] and metadata['height']:
                        metadata['size'] = f"{metadata['width']}x{metadata['height']}"

                except json.JSONDecodeError:
                    pass

    except Exception as e:
        print(f"Error parsing InvokeAI metadata: {e}")

    return metadata


def parse_novelai_metadata(raw_metadata):
    """Parse NovelAI metadata format"""
    metadata = {}

    try:
        # NovelAI specific fields
        if 'Description' in raw_metadata:
            # NovelAI often stores prompt in Description
            metadata['positive_prompt'] = raw_metadata['Description']

        if 'Comment' in raw_metadata:
            try:
                import json
                comment_data = json.loads(raw_metadata['Comment'])

                metadata['positive_prompt'] = comment_data.get('prompt', metadata.get('positive_prompt', ''))
                metadata['negative_prompt'] = comment_data.get('uc', '')  # NovelAI uses 'uc' for negative
                metadata['steps'] = comment_data.get('steps')
                metadata['sampler'] = comment_data.get('sampler')
                metadata['cfg_scale'] = comment_data.get('scale')
                metadata['seed'] = comment_data.get('seed')

            except json.JSONDecodeError:
                pass

    except Exception as e:
        print(f"Error parsing NovelAI metadata: {e}")

    return metadata


def parse_generic_metadata(raw_metadata):
    """Parse generic metadata formats"""
    metadata = {}

    try:
        # Map common field names
        field_mapping = {
            'prompt': 'positive_prompt',
            'positive': 'positive_prompt',
            'positive_prompt': 'positive_prompt',
            'negative': 'negative_prompt',
            'negative_prompt': 'negative_prompt',
            'steps': 'steps',
            'sampler': 'sampler',
            'cfg': 'cfg_scale',
            'cfg_scale': 'cfg_scale',
            'seed': 'seed',
            'width': 'width',
            'height': 'height',
            'model': 'model',
            'model_name': 'model'
        }

        for raw_key, mapped_key in field_mapping.items():
            if raw_key in raw_metadata:
                metadata[mapped_key] = raw_metadata[raw_key]

        # Create size field if width/height available
        if 'width' in metadata and 'height' in metadata:
            metadata['size'] = f"{metadata['width']}x{metadata['height']}"

    except Exception as e:
        print(f"Error parsing generic metadata: {e}")

    return metadata


def parse_json_metadata(raw_metadata):
    """Try to parse JSON from any metadata field"""
    metadata = {}

    try:
        import json

        for key, value in raw_metadata.items():
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    json_data = json.loads(value)

                    # If it's a dict, try to extract common fields
                    if isinstance(json_data, dict):
                        # Try common prompt field names
                        for prompt_key in ['prompt', 'positive_prompt', 'text']:
                            if prompt_key in json_data:
                                metadata['positive_prompt'] = json_data[prompt_key]
                                break

                        for neg_key in ['negative_prompt', 'negative', 'uc']:
                            if neg_key in json_data:
                                metadata['negative_prompt'] = json_data[neg_key]
                                break

                        # Extract other parameters
                        for param in ['steps', 'sampler', 'cfg_scale', 'seed', 'width', 'height']:
                            if param in json_data:
                                metadata[param] = json_data[param]

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"Error parsing JSON metadata: {e}")

    return metadata


def extract_image_metadata(image_path):
    """Extract metadata from various image formats"""
    file_ext = Path(image_path).suffix.lower()

    if file_ext == '.png':
        return extract_png_metadata(image_path)
    else:
        # For other formats, try to extract EXIF data and look for SD metadata
        metadata = {}
        try:
            with Image.open(image_path) as img:
                # Try to get EXIF data
                exif_data = img._getexif() or {}

                # CivitAI stores metadata in EXIF field 37510 (UserComment)
                if 37510 in exif_data:
                    metadata.update(parse_civitai_exif(exif_data[37510]))

                # Try other common EXIF fields for metadata
                for field_id, value in exif_data.items():
                    if isinstance(value, (str, bytes)) and len(str(value)) > 50:
                        # This might contain prompt data
                        parsed = try_parse_metadata_string(str(value))
                        if parsed:
                            metadata.update(parsed)

                # Add basic EXIF info
                metadata.update({str(k): str(v) for k, v in exif_data.items()})

        except Exception as e:
            print(f"Error extracting EXIF metadata: {e}")

        return metadata


def parse_civitai_exif(exif_data):
    """Parse CivitAI metadata from EXIF UserComment field (37510)"""
    metadata = {}

    try:
        # Check if we're dealing with string data (already stored in database)
        if isinstance(exif_data, str):
            print(f"DEBUG: Processing CivitAI EXIF data as STRING, length: {len(exif_data)}")
            print(f"DEBUG: String data starts with: {exif_data[:100]}...")
            
            # This is string data, use Method 13 directly
            if exif_data.startswith("b'UNICODE") and "\\x" in exif_data:
                print(f"DEBUG: Found string representation, using Method 13")
                
                # Remove the b' and ' wrapper
                content = exif_data[2:-1]
                
                # Skip past "UNICODE" to get to the actual hex data
                if content.startswith("UNICODE"):
                    # Find the first \x sequence after UNICODE
                    import re
                    first_hex = re.search(r'\\x[0-9a-fA-F]{2}', content)
                    if first_hex:
                        # Start from the first hex sequence
                        hex_content = content[first_hex.start():]
                        print(f"DEBUG: Extracted hex content: {hex_content[:100]}...")
                        
                        # Convert \x sequences to actual characters
                        def hex_to_char(match):
                            hex_val = match.group(1)
                            try:
                                char_code = int(hex_val, 16)
                                if char_code > 0:  # Skip null bytes
                                    return chr(char_code)
                                else:
                                    return ''  # Remove null bytes
                            except:
                                return match.group(0)
                        
                        # Convert all \x sequences
                        decoded_content = re.sub(r'\\x([0-9a-fA-F]{2})', hex_to_char, hex_content)
                        
                        # Clean up any remaining escape sequences
                        decoded_content = decoded_content.replace('\\n', '\n')
                        decoded_content = decoded_content.replace('\\"', '"')
                        decoded_content = decoded_content.replace('\\\\', '\\')
                        
                        text_data = decoded_content
                        print(f"DEBUG: Method 13 (string hex conversion) successful: {text_data[:100]}...")
                        
                        # Parse the decoded text
                        if text_data and len(text_data.strip()) > 10:
                            parsed_data = parse_civitai_text(text_data)
                            if parsed_data:
                                metadata.update(parsed_data)
                                print(f"DEBUG: Successfully parsed CivitAI metadata: {list(parsed_data.keys())}")
                            
                            # Always store the decoded text
                            metadata['decoded_civitai_text'] = text_data[:1000]
                            print(f"DEBUG: Stored decoded text for manual parsing: {text_data[:100]}...")
                            
                            return metadata
                
                # If we get here, the string parsing failed
                print(f"DEBUG: String parsing failed, storing as raw text")
                metadata['raw_metadata_text'] = exif_data[:1000]
                return metadata
        
        # CivitAI stores data as Unicode bytes in EXIF
        if isinstance(exif_data, bytes):
            print(f"DEBUG: Processing CivitAI EXIF data, type: {type(exif_data)}, length: {len(exif_data)}")
            print(f"DEBUG: First 100 bytes: {exif_data[:100]}")
            print(f"DEBUG: Data starts with: {str(exif_data)[:50]}...")
            
            # Try multiple decoding methods
            text_data = None

            # Method 1: Remove UNICODE header and decode UTF-16LE
            if exif_data.startswith(b'UNICODE\x00\x00'):
                try:
                    unicode_data = exif_data[10:]  # Skip UNICODE header
                    text_data = unicode_data.decode('utf-16le', errors='ignore')
                    
                    # Check if the decoded text is actually readable (not corrupted)
                    if text_data and len(text_data.strip()) > 10:
                        # Count non-ASCII characters to detect corruption
                        non_ascii_count = sum(1 for c in text_data[:200] if ord(c) > 127)
                        ascii_count = sum(1 for c in text_data[:200] if 32 <= ord(c) <= 126)
                        
                        if non_ascii_count > ascii_count * 2:  # If mostly non-ASCII, it's corrupted
                            print(f"DEBUG: Method 1 (UTF-16LE) produced corrupted text, rejecting")
                            text_data = None
                        else:
                            print(f"DEBUG: Method 1 (UTF-16LE) successful: {text_data[:100]}...")
                    else:
                        print(f"DEBUG: Method 1 (UTF-16LE) produced empty text, rejecting")
                        text_data = None
                except Exception as e:
                    print(f"DEBUG: Method 1 failed: {e}")
                    text_data = None
            
            # Method 1.5: NEW - Handle the specific format we're seeing (UNICODE with null bytes)
            if not text_data and exif_data.startswith(b'UNICODE'):
                try:
                    # This is the format: UNICODE\x00\x00{\x00"\x00...
                    # The data is UTF-8 encoded but with null bytes between each character
                    content = exif_data[7:]  # Remove 'UNICODE' prefix
                    
                    # Extract every other byte (skip the null bytes)
                    cleaned_data = b''
                    for i in range(0, len(content), 2):
                        if i < len(content):
                            cleaned_data += content[i:i+1]
                    
                    # Now decode as UTF-8
                    text_data = cleaned_data.decode('utf-8')
                    print(f"DEBUG: Method 1.5 (UNICODE null-byte format) successful: {text_data[:100]}...")
                except Exception as e:
                    print(f"DEBUG: Method 1.5 failed: {e}")

            # Method 2: Try UTF-8 decoding
            if not text_data:
                try:
                    text_data = exif_data.decode('utf-8', errors='ignore')
                    print(f"DEBUG: Method 2 (UTF-8) successful: {text_data[:100]}...")
                except Exception as e:
                    print(f"DEBUG: Method 2 failed: {e}")

            # Method 3: Try latin-1 then convert
            if not text_data:
                try:
                    text_data = exif_data.decode('latin-1', errors='ignore')
                    print(f"DEBUG: Method 3 (latin-1) successful: {text_data[:100]}...")
                except Exception as e:
                    print(f"DEBUG: Method 3 failed: {e}")

            # Method 4: Try removing null bytes and decoding
            if not text_data:
                try:
                    cleaned_data = exif_data.replace(b'\x00', b'')
                    text_data = cleaned_data.decode('utf-8', errors='ignore')
                    print(f"DEBUG: Method 4 (null-removed UTF-8) successful: {text_data[:100]}...")
                except Exception as e:
                    print(f"DEBUG: Method 4 failed: {e}")

            # Method 5: Manual Unicode conversion for CivitAI format
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # CivitAI uses UTF-16LE with null bytes between characters
                    if b'UNICODE\x00\x00' in exif_data:
                        start_pos = exif_data.find(b'UNICODE\x00\x00') + 10
                        unicode_bytes = exif_data[start_pos:]

                        # Convert every other byte (skip null bytes)
                        chars = []
                        for i in range(0, len(unicode_bytes), 2):
                            if i + 1 < len(unicode_bytes):
                                char_code = unicode_bytes[i] + (unicode_bytes[i + 1] << 8)
                                if 32 <= char_code <= 126 or char_code in [10, 13]:  # Printable ASCII + newlines
                                    chars.append(chr(char_code))
                                elif char_code == 0:
                                    continue
                        text_data = ''.join(chars)
                        print(f"DEBUG: Method 5 (manual UTF-16LE) successful: {text_data[:100]}...")
                except Exception as e:
                    print(f"DEBUG: Method 5 failed: {e}")

            # Method 6: Enhanced CivitAI parsing for the specific format shown
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # Handle the specific format: b'UNICODE\x00\x00...'
                    if exif_data.startswith(b"b'UNICODE\\x00\\x00"):
                        # Remove the b' prefix and UNICODE header
                        clean_data = exif_data[3:-1]  # Remove b' and '
                        if clean_data.startswith(b'UNICODE\\x00\\x00'):
                            # This is a string representation of bytes, need to decode it
                            clean_data = clean_data[10:]  # Skip UNICODE\\x00\\x00
                            
                            # Convert the escaped hex sequences back to actual bytes
                            try:
                                # Replace escaped hex sequences with actual bytes
                                import re
                                hex_pattern = r'\\x([0-9a-fA-F]{2})'
                                def hex_to_char(match):
                                    return chr(int(match.group(1), 16))
                                
                                clean_str = clean_data.decode('utf-8', errors='ignore')
                                clean_str = re.sub(hex_pattern, hex_to_char, clean_str)
                                text_data = clean_str
                                print(f"DEBUG: Method 6 (hex-unescaped) successful: {text_data[:100]}...")
                            except Exception as e:
                                print(f"DEBUG: Method 6 hex conversion failed: {e}")
                except Exception as e:
                    print(f"DEBUG: Method 6 failed: {e}")

            # Method 7: Handle the specific CivitAI format from debug output
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # The debug output shows the data is already a string representation
                    # Convert the bytes to string first to see what we're working with
                    debug_str = str(exif_data)
                    print(f"DEBUG: Method 7 examining string representation: {debug_str[:200]}...")
                    
                    # If it's a string representation of bytes, try to extract the actual content
                    if debug_str.startswith("b'") and debug_str.endswith("'"):
                        # Remove the b' and ' wrapper
                        content = debug_str[2:-1]
                        
                        # Handle the specific UNICODE\\x00\\x00 format
                        if content.startswith("UNICODE\\x00\\x00"):
                            # Skip the UNICODE header
                            content = content[10:]
                            
                            # Convert escaped hex sequences to actual characters
                            import re
                            hex_pattern = r'\\x([0-9a-fA-F]{2})'
                            def hex_to_char(match):
                                return chr(int(match.group(1), 16))
                            
                            text_data = re.sub(hex_pattern, hex_to_char, content)
                            print(f"DEBUG: Method 7 (string hex-unescaped) successful: {text_data[:100]}...")
                            
                except Exception as e:
                    print(f"DEBUG: Method 7 failed: {e}")

            # Method 8: Handle the specific CivitAI format with backslashes before every character
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # Convert bytes to string to examine the content
                    debug_str = str(exif_data)
                    print(f"DEBUG: Method 8 examining string representation: {debug_str[:200]}...")
                    
                    # Look for the pattern with backslashes before characters
                    if "\\\\1\\\\g\\\\i\\\\r\\\\l" in debug_str or "\\\\1\\\\g\\\\i\\\\r\\\\l" in str(exif_data):
                        # This is the specific CivitAI format with backslashes
                        # Extract the content and remove the backslashes
                        content = debug_str
                        
                        # Remove the b' and ' wrapper if present
                        if content.startswith("b'") and content.endswith("'"):
                            content = content[2:-1]
                        
                        # Remove the UNICODE header if present
                        if content.startswith("UNICODE\\\\x00\\\\x00"):
                            content = content[10:]
                        
                        # Remove backslashes before characters (but keep escaped quotes and newlines)
                        import re
                        # Replace \\" with " and \\n with \n first
                        content = content.replace('\\\\"', '"').replace('\\\\n', '\n')
                        # Then remove remaining backslashes before single characters
                        content = re.sub(r'\\\\([^"n])', r'\1', content)
                        
                        text_data = content
                        print(f"DEBUG: Method 8 (backslash-removed) successful: {text_data[:100]}...")
                        
                except Exception as e:
                    print(f"DEBUG: Method 8 failed: {e}")

            # Method 9: Nuclear option - just strip all backslashes and dump raw text
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # Convert to string and look for any content with lots of backslashes
                    debug_str = str(exif_data)
                    if debug_str.count('\\\\') > 10:  # If there are many backslashes
                        print(f"DEBUG: Method 9 (nuclear option) - found content with many backslashes")
                        
                        # Remove the b' and ' wrapper if present
                        content = debug_str
                        if content.startswith("b'") and content.endswith("'"):
                            content = content[2:-1]
                        
                        # Remove the UNICODE header if present
                        if content.startswith("UNICODE\\\\x00\\\\x00"):
                            content = content[10:]
                        
                        # Just strip ALL backslashes and dump the raw text
                        content = content.replace('\\\\', '')
                        
                        text_data = content
                        print(f"DEBUG: Method 9 (nuclear backslash removal) successful: {text_data[:100]}...")
                        print(f"DEBUG: This will be dumped as raw text in positive_prompt for manual parsing")
                        
                except Exception as e:
                    print(f"DEBUG: Method 9 failed: {e}")

            # Method 10: Handle string representation of bytes (like "b'UNICODE\\x00\\x00...'")
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # Convert to string and look for the specific pattern
                    debug_str = str(exif_data)
                    if debug_str.startswith("b'") and debug_str.endswith("'") and "\\\\x" in debug_str:
                        print(f"DEBUG: Method 10 (string bytes representation) - found pattern")
                        
                        # Remove the b' and ' wrapper
                        content = debug_str[2:-1]
                        
                        # Remove the UNICODE header if present
                        if content.startswith("UNICODE\\\\x00\\\\x00"):
                            content = content[10:]
                        
                        # Convert escaped hex sequences to actual characters
                        import re
                        # Replace \\x00 with actual null bytes, \\x01 with actual bytes, etc.
                        def hex_to_char(match):
                            hex_val = match.group(1)
                            try:
                                return chr(int(hex_val, 16))
                            except:
                                return match.group(0)
                        
                        # Convert all \\x sequences
                        content = re.sub(r'\\\\x([0-9a-fA-F]{2})', hex_to_char, content)
                        
                        # Also handle other escaped sequences
                        content = content.replace('\\\\n', '\n')
                        content = content.replace('\\\\"', '"')
                        content = content.replace('\\\\', '\\')
                        
                        text_data = content
                        print(f"DEBUG: Method 10 (hex conversion) successful: {text_data[:100]}...")
                        
                except Exception as e:
                    print(f"DEBUG: Method 10 failed: {e}")
            
            # Method 11: NEW - Handle the exact format we're seeing in the database
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # Convert to string and look for the exact pattern from the database
                    debug_str = str(exif_data)
                    if debug_str.startswith("b'UNICODE\\x00\\x00") and "\\x00" in debug_str:
                        print(f"DEBUG: Method 11 (exact database format) - found pattern")
                        
                        # Remove the b' and ' wrapper
                        content = debug_str[2:-1]
                        
                        # Remove the UNICODE header
                        if content.startswith("UNICODE\\x00\\x00"):
                            content = content[10:]
                        
                        # Convert the \x00 sequences to actual characters
                        import re
                        def hex_to_char(match):
                            hex_val = match.group(1)
                            try:
                                char_code = int(hex_val, 16)
                                if char_code > 0:  # Skip null bytes
                                    return chr(char_code)
                                else:
                                    return ''  # Remove null bytes
                            except:
                                return match.group(0)
                        
                        # Convert all \x sequences (single backslash, not double)
                        content = re.sub(r'\\x([0-9a-fA-F]{2})', hex_to_char, content)
                        
                        # Clean up any remaining escape sequences
                        content = content.replace('\\n', '\n')
                        content = content.replace('\\"', '"')
                        content = content.replace('\\', '')
                        
                        text_data = content
                        print(f"DEBUG: Method 11 (exact format conversion) successful: {text_data[:100]}...")
                        
                except Exception as e:
                    print(f"DEBUG: Method 11 failed: {e}")
            
            # Method 12: NEW - Use JavaScript-style unescape approach (like magictool.ai)
            if not text_data or len(text_data.strip()) < 50:
                try:
                    # Convert to string and look for the pattern
                    debug_str = str(exif_data)
                    if debug_str.startswith("b'UNICODE") and "\\x" in debug_str:
                        print(f"DEBUG: Method 12 (JavaScript unescape approach) - found pattern")
                        
                        # Remove the b' and ' wrapper
                        content = debug_str[2:-1]
                        
                        # Remove the UNICODE header (keep the rest)
                        if content.startswith("UNICODE"):
                            # Find where the actual content starts after UNICODE
                            # Look for the first \x sequence
                            import re
                            first_hex = re.search(r'\\x[0-9a-fA-F]{2}', content)
                            if first_hex:
                                content = content[first_hex.start():]
                                print(f"DEBUG: Removed UNICODE prefix, content starts with: {content[:50]}...")
                        
                        # Now use a JavaScript-style unescape approach
                        # Convert \x sequences to actual characters
                        def hex_to_char(match):
                            hex_val = match.group(1)
                            try:
                                char_code = int(hex_val, 16)
                                return chr(char_code)
                            except:
                                return match.group(0)
                        
                        # Convert all \x sequences
                        content = re.sub(r'\\x([0-9a-fA-F]{2})', hex_to_char, content)
                        
                        # Clean up any remaining escape sequences
                        content = content.replace('\\n', '\n')
                        content = content.replace('\\"', '"')
                        content = content.replace('\\\\', '\\')
                        
                        text_data = content
                        print(f"DEBUG: Method 12 (JavaScript unescape) successful: {text_data[:100]}...")
                        
                except Exception as e:
                    print(f"DEBUG: Method 12 failed: {e}")
            
            # Method 13: NEW - Handle string representation that's already in the database
            # This should ALWAYS run for string data, regardless of previous methods
            if isinstance(exif_data, str) and exif_data.startswith("b'UNICODE") and "\\x" in exif_data:
                try:
                    # This handles the case where the data is already stored as a string
                    debug_str = str(exif_data)
                    if debug_str.startswith("b'UNICODE") and "\\x" in debug_str:
                        print(f"DEBUG: Method 13 (string representation) - found pattern")
                        
                        # Remove the b' and ' wrapper
                        content = debug_str[2:-1]
                        
                        # Skip past "UNICODE" to get to the actual hex data
                        if content.startswith("UNICODE"):
                            # Find the first \x sequence after UNICODE
                            import re
                            first_hex = re.search(r'\\x[0-9a-fA-F]{2}', content)
                            if first_hex:
                                # Start from the first hex sequence
                                hex_content = content[first_hex.start():]
                                print(f"DEBUG: Extracted hex content: {hex_content[:100]}...")
                                
                                # Convert \x sequences to actual characters
                                def hex_to_char(match):
                                    hex_val = match.group(1)
                                    try:
                                        char_code = int(hex_val, 16)
                                        if char_code > 0:  # Skip null bytes
                                            return chr(char_code)
                                        else:
                                            return ''  # Remove null bytes
                                    except:
                                        return match.group(0)
                                
                                # Convert all \x sequences
                                decoded_content = re.sub(r'\\x([0-9a-fA-F]{2})', hex_to_char, hex_content)
                                
                                # Clean up any remaining escape sequences
                                decoded_content = decoded_content.replace('\\n', '\n')
                                decoded_content = decoded_content.replace('\\"', '"')
                                decoded_content = decoded_content.replace('\\\\', '\\')
                                
                                text_data = decoded_content
                                print(f"DEBUG: Method 13 (string hex conversion) successful: {text_data[:100]}...")
                        
                except Exception as e:
                    print(f"DEBUG: Method 13 failed: {e}")

            # Fallback: just convert bytes to string
            if not text_data:
                text_data = str(exif_data)
                print(f"DEBUG: Using fallback string conversion: {text_data[:100]}...")

        else:
            text_data = str(exif_data)

        # Clean up the text data
        if text_data:
            # Remove null characters and clean up
            text_data = text_data.replace('\x00', '').strip()

            # If it still looks corrupted, try one more approach
            if any(ord(c) > 127 for c in text_data[:100]) and 'score_' not in text_data:
                # This might be double-encoded, try to extract ASCII parts
                ascii_chars = []
                for char in text_data:
                    if 32 <= ord(char) <= 126 or char in ['\n', '\r', '\t']:
                        ascii_chars.append(char)
                text_data = ''.join(ascii_chars)

        print(f"CivitAI decoded text (first 200 chars): {text_data[:200]}...")

        # Parse the CivitAI format
        if text_data and len(text_data.strip()) > 10:
            parsed_data = parse_civitai_text(text_data)
            if parsed_data:
                metadata.update(parsed_data)
                print(f"DEBUG: Successfully parsed CivitAI metadata: {list(parsed_data.keys())}")
            else:
                # If parsing fails, store the raw decoded text as fallback
                metadata['raw_metadata_text'] = text_data[:1000]  # Limit length
                print(f"DEBUG: Parsing failed, storing raw text as fallback")
        else:
            # If no text data, store the original exif data as fallback
            metadata['raw_metadata_text'] = str(exif_data)[:1000]  # Limit length
            print(f"DEBUG: No text data, storing original exif data as fallback")
        
        # ALWAYS store the decoded text for debugging and manual parsing
        if text_data and len(text_data.strip()) > 10:
            metadata['decoded_civitai_text'] = text_data[:1000]  # Store the clean decoded text
            print(f"DEBUG: Stored decoded text for manual parsing: {text_data[:100]}...")
            
            # Also try to extract a basic prompt if parsing failed
            if not metadata.get('positive_prompt') and not metadata.get('negative_prompt'):
                # Look for common prompt patterns in the decoded text
                decoded_text = text_data.lower()
                if any(marker in decoded_text for marker in ['girl', 'boy', 'woman', 'man', 'masterpiece', 'best quality']):
                    # Extract the first part as positive prompt
                    lines = text_data.split('\n')
                    if lines:
                        first_line = lines[0].strip()
                        if len(first_line) > 10:  # Only if it's substantial
                            metadata['positive_prompt'] = first_line
                            print(f"DEBUG: Extracted basic prompt from decoded text: {first_line[:100]}...")

    except Exception as e:
        print(f"Error parsing CivitAI EXIF: {e}")

    return metadata


def auto_detect_model_name(metadata):
    """Auto-detect model name from metadata if not provided by user"""
    model_candidates = []

    # Check various metadata fields for model information
    for field in ['model', 'model_name', 'base_model', 'checkpoint']:
        if field in metadata and metadata[field]:
            model_candidates.append(metadata[field])

    # Extract from CivitAI resources
    if 'base_model' in metadata:
        model_candidates.append(metadata['base_model'])

    # Clean and return the best candidate
    for candidate in model_candidates:
        if candidate and len(str(candidate).strip()) > 3:
            # Clean up model name
            clean_name = str(candidate).strip()
            # Remove file extensions
            clean_name = re.sub(r'\.(ckpt|safetensors|pt|bin)$', '', clean_name, flags=re.IGNORECASE)
            # Remove common path separators
            clean_name = clean_name.split('/')[-1].split('\\')[-1]
            return clean_name

    return None


def parse_civitai_text(text_data):
    """Parse CivitAI text format"""
    metadata = {}

    try:
        print(f"DEBUG: parse_civitai_text called with: {text_data[:200]}...")
        
        # First, try to parse as structured format with markers
        lines = text_data.split('\n')
        current_section = "positive_prompt"
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for negative prompt marker
            if line.startswith('Negative prompt:'):
                # Save current section
                if current_text:
                    metadata[current_section] = '\n'.join(current_text).strip()
                    current_text = []

                # Start negative prompt section
                current_section = "negative_prompt"
                # Extract text after "Negative prompt:"
                neg_text = line.replace('Negative prompt:', '').strip()
                if neg_text:
                    current_text.append(neg_text)
                continue

            # Check for parameters section
            if any(param in line for param in ['Steps:', 'Sampler:', 'CFG scale:', 'Seed:', 'Size:']):
                # Save current section
                if current_text:
                    metadata[current_section] = '\n'.join(current_text).strip()
                    current_text = []

                # Parse parameters
                metadata.update(parse_parameters_line(line))
                continue

            # Check for CivitAI resources
            if 'Civitai resources:' in line:
                # Save current section
                if current_text:
                    metadata[current_section] = '\n'.join(current_text).strip()
                    current_text = []

                # Extract resources JSON
                resources_start = line.find('[')
                if resources_start != -1:
                    resources_json = line[resources_start:]
                    metadata.update(parse_civitai_resources(resources_json))
                continue

            # Check for CivitAI metadata
            if 'Civitai metadata:' in line:
                metadata_start = line.find('{')
                if metadata_start != -1:
                    civitai_meta = line[metadata_start:]
                    try:
                        import json
                        meta_data = json.loads(civitai_meta)
                        metadata['civitai_remix_id'] = meta_data.get('remixOfId')
                    except:
                        pass
                continue

            # Add to current section
            current_text.append(line)

        # Save final section
        if current_text:
            metadata[current_section] = '\n'.join(current_text).strip()
        
        # If we didn't get any structured data, try to extract basic prompt info
        if not metadata.get('positive_prompt') and not metadata.get('negative_prompt'):
            print(f"DEBUG: No structured data found, trying basic prompt extraction")
            
            # Look for common prompt patterns
            text_lower = text_data.lower()
            if any(marker in text_lower for marker in ['girl', 'boy', 'woman', 'man', 'masterpiece', 'best quality']):
                # Split by newlines and take the first substantial line
                lines = text_data.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 10 and any(marker in line.lower() for marker in ['girl', 'boy', 'woman', 'man', 'masterpiece', 'best quality']):
                        metadata['positive_prompt'] = line
                        print(f"DEBUG: Extracted basic prompt: {line[:100]}...")
                        break
                
                # If we still don't have a prompt, just take the first line
                if not metadata.get('positive_prompt') and lines:
                    first_line = lines[0].strip()
                    if len(first_line) > 10:
                        metadata['positive_prompt'] = first_line
                        print(f"DEBUG: Using first line as prompt: {first_line[:100]}...")

    except Exception as e:
        print(f"Error parsing CivitAI text: {e}")

    print(f"DEBUG: parse_civitai_text returning: {list(metadata.keys())}")
    return metadata


def parse_parameters_line(params_line):
    """Parse a line containing generation parameters"""
    metadata = {}

    try:
        # Extract individual parameters using regex
        param_pattern = r'(\w+(?:\s+\w+)*?):\s*([^,]+?)(?=,\s*\w+(?:\s+\w+)*?:|$)'
        param_pairs = re.findall(param_pattern, params_line)

        for param_name, param_value in param_pairs:
            clean_name = param_name.strip().lower().replace(' ', '_')
            clean_value = param_value.strip()

            # Handle specific parameter types
            if clean_name in ['steps', 'cfg_scale', 'seed']:
                try:
                    metadata[clean_name] = float(clean_value) if '.' in clean_value else int(clean_value)
                except:
                    metadata[clean_name] = clean_value
            elif clean_name == 'size':
                metadata[clean_name] = clean_value
                # Also extract width/height
                if 'x' in clean_value:
                    try:
                        width, height = clean_value.split('x')
                        metadata['width'] = int(width.strip())
                        metadata['height'] = int(height.strip())
                    except:
                        pass
            elif clean_name == 'created_date':
                metadata['creation_date'] = clean_value
            else:
                metadata[clean_name] = clean_value

    except Exception as e:
        print(f"Error parsing parameters line: {e}")

    return metadata


def parse_civitai_resources(resources_json):
    """Parse CivitAI resources JSON to extract model information"""
    metadata = {}

    try:
        import json
        resources = json.loads(resources_json)

        models = []
        loras = []
        embeddings = []

        for resource in resources:
            resource_type = resource.get('type', '')
            model_name = resource.get('modelName', '')
            version_name = resource.get('modelVersionName', '')
            weight = resource.get('weight', 1)

            if resource_type == 'checkpoint':
                models.append(f"{model_name} ({version_name})")
                metadata['model'] = f"{model_name}"
                metadata['model_version'] = version_name
            elif resource_type == 'lora':
                loras.append(f"{model_name} ({weight})")
            elif resource_type in ['embed', 'embedding']:
                embeddings.append(model_name)
            elif resource_type == 'vae':
                metadata['vae'] = model_name

        if models:
            metadata['base_model'] = models[0] if len(models) == 1 else ', '.join(models)
        if loras:
            metadata['loras'] = ', '.join(loras)
        if embeddings:
            metadata['embeddings'] = ', '.join(embeddings)

    except Exception as e:
        print(f"Error parsing CivitAI resources: {e}")

    return metadata


def try_parse_metadata_string(text):
    """Try to parse any string that might contain SD metadata"""
    metadata = {}

    try:
        # Look for common patterns
        if 'Negative prompt:' in text and any(x in text for x in ['Steps:', 'Sampler:', 'CFG']):
            # Looks like AUTOMATIC1111 format
            metadata.update(parse_a1111_parameters(text))
        elif 'Civitai' in text:
            # CivitAI format
            metadata.update(parse_civitai_text(text))
        elif text.startswith('{') or text.startswith('['):
            # JSON format
            try:
                import json
                json_data = json.loads(text)
                metadata.update(parse_json_metadata({'data': json_data}))
            except:
                pass

    except Exception as e:
        print(f"Error parsing metadata string: {e}")

    return metadata


@app.route('/')
def index():
    """Main gallery page with pagination"""
    db = load_database()

    # Get filter and pagination parameters
    model_filter = request.args.get('model', '').strip()
    tag_filter = request.args.get('tag', '').strip()
    search_term = request.args.get('search', '').strip()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 30))  # Default 30 images per page

    # Filter images
    filtered_images = db

    if model_filter:
        filtered_images = [img for img in filtered_images if model_filter.lower() in img.get('model_name', '').lower()]

    if tag_filter:
        filtered_images = [img for img in filtered_images if
                           tag_filter.lower() in ' '.join(img.get('content_tags', [])).lower()]

    if search_term:
        filtered_images = [img for img in filtered_images if
                           search_term.lower() in img.get('positive_prompt', '').lower() or
                           search_term.lower() in img.get('negative_prompt', '').lower() or
                           search_term.lower() in img.get('filename', '').lower()]

    # Calculate pagination
    total_images = len(filtered_images)
    total_pages = (total_images + per_page - 1) // per_page  # Ceiling division
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_images = filtered_images[start_idx:end_idx]

    # Calculate pagination info
    has_prev = page > 1
    has_next = page < total_pages
    prev_page = page - 1 if has_prev else None
    next_page = page + 1 if has_next else None

    # Get page range for pagination links (show 5 pages around current)
    page_range_start = max(1, page - 2)
    page_range_end = min(total_pages + 1, page + 3)
    page_range = list(range(page_range_start, page_range_end))

    # Get unique models and tags for filter dropdowns
    all_models = list(set(img.get('model_name', '') for img in db if img.get('model_name')))
    all_tags = list(set(tag for img in db for tag in img.get('content_tags', [])))

    return render_template('gallery.html',
                           images=paginated_images,
                           models=sorted(all_models),
                           tags=sorted(all_tags),
                           current_model=model_filter,
                           current_tag=tag_filter,
                           current_search=search_term,
                           pagination={
                               'page': page,
                               'per_page': per_page,
                               'total': total_images,
                               'total_pages': total_pages,
                               'has_prev': has_prev,
                               'has_next': has_next,
                               'prev_page': prev_page,
                               'next_page': next_page,
                               'page_range': page_range
                           })


@app.route('/image/<filename>')
def get_image(filename):
    """Serve image files"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/image-detail/<int:image_id>')
def image_detail(image_id):
    """Show detailed view of a specific image"""
    db = load_database()

    if 0 <= image_id < len(db):
        image = db[image_id]
        return render_template('image_detail.html', image=image, image_id=image_id)
    else:
        flash('Image not found', 'error')
        return redirect(url_for('index'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    """Handle single or multiple image upload and metadata extraction"""
    if request.method == 'POST':
        files = request.files.getlist('files')  # Changed to handle multiple files
        model_name = request.form.get('model_name', '').strip()
        content_tags = [tag.strip() for tag in request.form.get('content_tags', '').split(',') if tag.strip()]

        if not files or not any(f.filename for f in files):
            flash('No files selected', 'error')
            return redirect(request.url)

        successful_uploads = 0
        failed_uploads = 0
        duplicate_uploads = 0
        upload_results = []

        for file in files:
            if not file.filename:
                continue

            if not allowed_file(file.filename):
                upload_results.append(f"❌ {file.filename}: Invalid file type")
                failed_uploads += 1
                continue

            # Create a temporary file to process
            temp_path = os.path.join(UPLOAD_FOLDER, 'temp_' + file.filename)

            try:
                file.save(temp_path)

                # Generate hash to check for duplicates
                file_hash = generate_file_hash(temp_path)

                # Check if image already exists
                db = load_database()
                existing_image = next((img for img in db if img.get('file_hash') == file_hash), None)

                if existing_image:
                    os.remove(temp_path)
                    upload_results.append(f"⚠️ {file.filename}: Already exists as {existing_image['filename']}")
                    duplicate_uploads += 1
                    continue

                # Extract metadata
                metadata = extract_image_metadata(temp_path)

                # Auto-detect model name if not provided by user
                final_model_name = model_name
                if not final_model_name:
                    detected_model = auto_detect_model_name(metadata)
                    if detected_model:
                        final_model_name = detected_model

                # Create final filename (timestamp + original name to avoid conflicts)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include microseconds for uniqueness
                final_filename = f"{timestamp}_{file.filename}"
                final_path = os.path.join(UPLOAD_FOLDER, final_filename)

                # Move file to final location
                shutil.move(temp_path, final_path)

                # Clean up metadata - remove raw EXIF fields that weren't parsed
                cleaned_metadata = {}
                for key, value in metadata.items():
                    # Keep parsed fields
                    if key in ['positive_prompt', 'negative_prompt', 'steps', 'sampler', 'cfg_scale', 'seed', 'size', 'model_name', 'raw_metadata_text']:
                        cleaned_metadata[key] = value
                    # Keep other useful parsed fields
                    elif not key.isdigit() and not key.startswith('EXIF_'):
                        cleaned_metadata[key] = value
                    # For raw metadata text, store it cleanly
                    elif key == 'raw_metadata_text':
                        cleaned_metadata[key] = value

                # Create database entry
                image_data = {
                    'filename': final_filename,
                    'original_filename': file.filename,
                    'upload_date': datetime.now().isoformat(),
                    'model_name': final_model_name,
                    'content_tags': content_tags,
                    'file_hash': file_hash,
                    'file_size': os.path.getsize(final_path),
                    **cleaned_metadata  # Include only cleaned metadata
                }

                # Add to database
                db.append(image_data)
                save_database(db)

                model_info = f" (Model: {final_model_name})" if final_model_name else ""
                upload_results.append(f"✅ {file.filename}: Uploaded successfully{model_info}")
                successful_uploads += 1

            except Exception as e:
                # Clean up on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                upload_results.append(f"❌ {file.filename}: Error - {str(e)}")
                failed_uploads += 1

        # Display summary
        if successful_uploads > 0:
            flash(f'Successfully uploaded {successful_uploads} file(s)', 'success')
        if duplicate_uploads > 0:
            flash(f'{duplicate_uploads} duplicate file(s) skipped', 'warning')
        if failed_uploads > 0:
            flash(f'{failed_uploads} file(s) failed to upload', 'error')

        # Display detailed results
        for result in upload_results[:10]:  # Limit to first 10 results to avoid UI overflow
            if result.startswith('✅'):
                flash(result, 'success')
            elif result.startswith('⚠️'):
                flash(result, 'warning')
            else:
                flash(result, 'error')

        if len(upload_results) > 10:
            flash(f'... and {len(upload_results) - 10} more results', 'info')

        return redirect(url_for('index'))

    return render_template('upload.html')


@app.route('/api/images')
def api_images():
    """API endpoint for images (for future AJAX functionality)"""
    db = load_database()
    return jsonify(db)


@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """Delete an image and its database entry"""
    db = load_database()

    if 0 <= image_id < len(db):
        image = db[image_id]
        image_path = os.path.join(UPLOAD_FOLDER, image['filename'])

        # Remove file if it exists
        if os.path.exists(image_path):
            os.remove(image_path)

        # Remove from database
        db.pop(image_id)
        save_database(db)

        flash('Image deleted successfully', 'success')
    else:
        flash('Image not found', 'error')

    return redirect(url_for('index'))


@app.route('/reprocess-metadata')
def reprocess_metadata():
    """Reprocess all existing images in the database with updated metadata parsing"""
    db = load_database()
    updated_count = 0
    cleaned_count = 0
    
    for i, image in enumerate(db):
        # Process all images, not just those with raw EXIF fields
        print(f"Reprocessing metadata for image {i}: {image.get('filename', 'unknown')}")
        
        # Extract metadata from the image file
        image_path = os.path.join(UPLOAD_FOLDER, image.get('filename', ''))
        if os.path.exists(image_path):
            try:
                # Extract metadata using our updated functions
                extracted_metadata = extract_png_metadata(image_path)
                
                # Clean up the image data - remove raw EXIF fields if they exist
                keys_to_remove = []
                for key in image.keys():
                    if key.isdigit() or key.startswith('EXIF_'):
                        keys_to_remove.append(key)
                
                # Remove raw EXIF fields
                for key in keys_to_remove:
                    del image[key]
                    cleaned_count += 1
                
                # Update the image with extracted metadata
                if extracted_metadata:
                    # Preserve existing fields but update with new parsed data
                    for key, value in extracted_metadata.items():
                        if key not in ['filename', 'original_filename', 'upload_date', 'file_hash', 'file_size']:
                            image[key] = value
                    
                    updated_count += 1
                    print(f"  Updated metadata for {image.get('filename')}")
                    
                    # Debug output
                    if 'positive_prompt' in extracted_metadata:
                        print(f"    Positive prompt: {extracted_metadata['positive_prompt'][:100]}...")
                    if 'raw_metadata_text' in extracted_metadata:
                        print(f"    Raw metadata: {extracted_metadata['raw_metadata_text'][:100]}...")
                    
                    # Also show what was extracted
                    print(f"    Extracted fields: {list(extracted_metadata.keys())}")
                    
                else:
                    print(f"  No metadata extracted from {image.get('filename')}")
                    
            except Exception as e:
                print(f"  Error reprocessing {image.get('filename')}: {e}")
    
    # Save the updated database
    if updated_count > 0 or cleaned_count > 0:
        save_database(db)
        flash(f'Successfully reprocessed metadata for {updated_count} images and cleaned {cleaned_count} raw EXIF fields', 'success')
    else:
        flash('No images needed metadata reprocessing', 'info')
    
    return redirect(url_for('index'))


@app.route('/clean-database')
def clean_database():
    """Clean up raw EXIF fields from the database without reprocessing"""
    db = load_database()
    cleaned_count = 0
    
    for i, image in enumerate(db):
        keys_to_remove = []
        for key in image.keys():
            if key.isdigit() or key.startswith('EXIF_'):
                keys_to_remove.append(key)
        
        if keys_to_remove:
            print(f"Cleaning image {i}: {image.get('filename', 'unknown')} - removing {len(keys_to_remove)} raw fields")
            for key in keys_to_remove:
                del image[key]
                cleaned_count += 1
    
    # Save the cleaned database
    if cleaned_count > 0:
        save_database(db)
        flash(f'Successfully cleaned {cleaned_count} raw EXIF fields from the database', 'success')
    else:
        flash('No raw EXIF fields found to clean', 'info')
    
    return redirect(url_for('index'))


@app.route('/debug-metadata/<filename>')
def debug_metadata(filename):
    """Debug route to show all available metadata in an image"""
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    debug_info = {
        'filename': filename,
        'raw_metadata': {},
        'parsed_metadata': {},
        'file_info': {},
        'exif_data': {},
        'special_fields': {},
        'civitai_parsing_test': {}
    }

    try:
        # Get file info
        stat = os.stat(image_path)
        debug_info['file_info'] = {
            'size_bytes': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }

        # Extract raw metadata
        with Image.open(image_path) as img:
            debug_info['file_info']['format'] = img.format
            debug_info['file_info']['mode'] = img.mode
            debug_info['file_info']['size'] = img.size

            # PNG text chunks
            if hasattr(img, 'text'):
                debug_info['raw_metadata'] = dict(img.text)

            # EXIF data for all formats
            try:
                exif_data = img._getexif() or {}
                debug_info['exif_data'] = {}

                for key, value in exif_data.items():
                    if key == 37510:  # CivitAI UserComment field
                        debug_info['special_fields']['civitai_usercomment_raw'] = repr(value)

                        # Try to decode CivitAI data
                        if isinstance(value, bytes) and value.startswith(b'UNICODE\x00\x00'):
                            try:
                                unicode_data = value[10:]
                                decoded = unicode_data.decode('utf-16le', errors='ignore')
                                debug_info['special_fields']['civitai_decoded'] = decoded
                            except Exception as decode_err:
                                debug_info['special_fields']['civitai_decode_error'] = str(decode_err)

                    # Store all EXIF data with readable names when possible
                    key_name = f"EXIF_{key}"
                    if isinstance(value, bytes) and len(value) > 100:
                        debug_info['exif_data'][key_name] = f"<bytes data, length: {len(value)}>"
                    else:
                        debug_info['exif_data'][key_name] = str(value)

            except Exception as exif_error:
                debug_info['exif_error'] = str(exif_error)

        # Get parsed metadata using our extraction function
        debug_info['parsed_metadata'] = extract_image_metadata(image_path)
        
        # Test CivitAI parsing specifically
        try:
            # Look for EXIF field 37510 (CivitAI UserComment)
            if 'exif_data' in debug_info and 'EXIF_37510' in debug_info['exif_data']:
                raw_data = debug_info['exif_data']['EXIF_37510']
                debug_info['civitai_parsing_test']['raw_data'] = str(raw_data)
                
                # Test our parsing methods
                if isinstance(raw_data, bytes):
                    # Test the parse_civitai_exif function
                    parsed = parse_civitai_exif(raw_data)
                    debug_info['civitai_parsing_test']['parsed_result'] = parsed
                    debug_info['civitai_parsing_test']['parsing_successful'] = bool(parsed.get('positive_prompt') or parsed.get('raw_metadata_text'))
                else:
                    debug_info['civitai_parsing_test']['error'] = 'Data is not bytes, cannot parse with parse_civitai_exif'
                    
        except Exception as e:
            debug_info['civitai_parsing_test']['error'] = str(e)

    except Exception as e:
        debug_info['error'] = str(e)

    return jsonify(debug_info)


@app.route('/debug')
def debug_page():
    """Debug page to test metadata extraction"""
    db = load_database()
    return render_template('debug.html', images=db)


@app.route('/stats')
def stats():
    """Show statistics about the gallery"""
    db = load_database()

    if not db:
        return render_template('stats.html', stats={})

    # Calculate statistics
    total_images = len(db)
    models = {}
    tags = {}
    total_size = 0
    metadata_sources = {}  # Track where metadata came from

    for img in db:
        # Model statistics
        model = img.get('model_name', 'Unknown')
        models[model] = models.get(model, 0) + 1

        # Tag statistics
        for tag in img.get('content_tags', []):
            tags[tag] = tags.get(tag, 0) + 1

        # File size
        total_size += img.get('file_size', 0)

        # Track metadata sources
        has_prompts = bool(img.get('positive_prompt') or img.get('negative_prompt'))
        has_params = bool(img.get('steps') or img.get('sampler'))

        if has_prompts and has_params:
            metadata_sources['Full Metadata'] = metadata_sources.get('Full Metadata', 0) + 1
        elif has_prompts:
            metadata_sources['Prompts Only'] = metadata_sources.get('Prompts Only', 0) + 1
        elif has_params:
            metadata_sources['Parameters Only'] = metadata_sources.get('Parameters Only', 0) + 1
        else:
            metadata_sources['No Metadata'] = metadata_sources.get('No Metadata', 0) + 1

    stats = {
        'total_images': total_images,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'models': dict(sorted(models.items(), key=lambda x: x[1], reverse=True)),
        'tags': dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)),
        'avg_file_size_mb': round((total_size / total_images) / (1024 * 1024), 2) if total_images > 0 else 0,
        'metadata_sources': metadata_sources
    }

    return render_template('stats.html', stats=stats)


@app.route('/edit/<int:image_id>', methods=['GET', 'POST'])
def edit_image(image_id):
    """Edit image metadata"""
    db = load_database()

    if not (0 <= image_id < len(db)):
        flash('Image not found', 'error')
        return redirect(url_for('index'))

    image = db[image_id]

    if request.method == 'POST':
        # Update metadata
        image['model_name'] = request.form.get('model_name', '').strip()

        # Update content tags
        content_tags = request.form.get('content_tags', '').strip()
        image['content_tags'] = [tag.strip() for tag in content_tags.split(',') if tag.strip()]

        # Update prompts if provided
        positive_prompt = request.form.get('positive_prompt', '').strip()
        if positive_prompt:
            image['positive_prompt'] = positive_prompt

        negative_prompt = request.form.get('negative_prompt', '').strip()
        if negative_prompt:
            image['negative_prompt'] = negative_prompt

        # Save changes
        save_database(db)
        flash('Image metadata updated successfully', 'success')
        return redirect(url_for('image_detail', image_id=image_id))

    return render_template('edit_image.html', image=image, image_id=image_id)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
