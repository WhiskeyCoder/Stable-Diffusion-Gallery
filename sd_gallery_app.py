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
                metadata.update(raw_metadata)

                # Try different metadata formats
                parsed_data = None

                # Method 1: AUTOMATIC1111 format (parameters field)
                if 'parameters' in raw_metadata:
                    parsed_data = parse_a1111_parameters(raw_metadata['parameters'])

                # Method 2: ComfyUI format (workflow/prompt fields)
                elif 'workflow' in raw_metadata or 'prompt' in raw_metadata:
                    parsed_data = parse_comfyui_metadata(raw_metadata)

                # Method 3: InvokeAI format
                elif 'invokeai_metadata' in raw_metadata or 'sd-metadata' in raw_metadata:
                    parsed_data = parse_invokeai_metadata(raw_metadata)

                # Method 4: NovelAI format
                elif 'Software' in raw_metadata and 'NovelAI' in raw_metadata.get('Software', ''):
                    parsed_data = parse_novelai_metadata(raw_metadata)

                # Method 5: Generic prompt fields
                elif any(key in raw_metadata for key in ['prompt', 'positive', 'negative']):
                    parsed_data = parse_generic_metadata(raw_metadata)

                # Method 6: Check for JSON in any field
                else:
                    parsed_data = parse_json_metadata(raw_metadata)

                # Merge parsed data
                if parsed_data:
                    metadata.update(parsed_data)

                # Debug: Print all available metadata keys for troubleshooting
                print(f"Available metadata keys: {list(raw_metadata.keys())}")

    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")

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
        # CivitAI stores data as Unicode bytes in EXIF
        if isinstance(exif_data, bytes):
            # Try multiple decoding methods
            text_data = None

            # Method 1: Remove UNICODE header and decode UTF-16LE
            if exif_data.startswith(b'UNICODE\x00\x00'):
                try:
                    unicode_data = exif_data[10:]  # Skip UNICODE header
                    text_data = unicode_data.decode('utf-16le', errors='ignore')
                except:
                    pass

            # Method 2: Try UTF-8 decoding
            if not text_data:
                try:
                    text_data = exif_data.decode('utf-8', errors='ignore')
                except:
                    pass

            # Method 3: Try latin-1 then convert
            if not text_data:
                try:
                    text_data = exif_data.decode('latin-1', errors='ignore')
                except:
                    pass

            # Method 4: Try removing null bytes and decoding
            if not text_data:
                try:
                    cleaned_data = exif_data.replace(b'\x00', b'')
                    text_data = cleaned_data.decode('utf-8', errors='ignore')
                except:
                    pass

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
                except:
                    pass

            # Fallback: just convert bytes to string
            if not text_data:
                text_data = str(exif_data)

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
            metadata.update(parse_civitai_text(text_data))

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
        # Split by newlines and parse each section
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

    except Exception as e:
        print(f"Error parsing CivitAI text: {e}")

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
    """Main gallery page"""
    db = load_database()

    # Get filter parameters
    model_filter = request.args.get('model', '').strip()
    tag_filter = request.args.get('tag', '').strip()
    search_term = request.args.get('search', '').strip()

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

    # Get unique models and tags for filter dropdowns
    all_models = list(set(img.get('model_name', '') for img in db if img.get('model_name')))
    all_tags = list(set(tag for img in db for tag in img.get('content_tags', [])))

    return render_template('gallery.html',
                           images=filtered_images,
                           models=sorted(all_models),
                           tags=sorted(all_tags),
                           current_model=model_filter,
                           current_tag=tag_filter,
                           current_search=search_term)


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
    """Handle image upload and metadata extraction"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        model_name = request.form.get('model_name', '').strip()
        content_tags = [tag.strip() for tag in request.form.get('content_tags', '').split(',') if tag.strip()]

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Create a temporary file to process
            temp_path = os.path.join(UPLOAD_FOLDER, 'temp_' + file.filename)
            file.save(temp_path)

            try:
                # Generate hash to check for duplicates
                file_hash = generate_file_hash(temp_path)

                # Check if image already exists
                db = load_database()
                existing_image = next((img for img in db if img.get('file_hash') == file_hash), None)

                if existing_image:
                    os.remove(temp_path)
                    flash(f'Image already exists: {existing_image["filename"]}', 'warning')
                    return redirect(url_for('index'))

                # Extract metadata
                metadata = extract_image_metadata(temp_path)

                # Auto-detect model name if not provided by user
                final_model_name = model_name
                if not final_model_name:
                    detected_model = auto_detect_model_name(metadata)
                    if detected_model:
                        final_model_name = detected_model
                        flash(f'Auto-detected model: {detected_model}', 'info')

                # Create final filename (timestamp + original name to avoid conflicts)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                final_filename = f"{timestamp}_{file.filename}"
                final_path = os.path.join(UPLOAD_FOLDER, final_filename)

                # Move file to final location
                shutil.move(temp_path, final_path)

                # Create database entry
                image_data = {
                    'filename': final_filename,
                    'original_filename': file.filename,
                    'upload_date': datetime.now().isoformat(),
                    'model_name': final_model_name,
                    'content_tags': content_tags,
                    'file_hash': file_hash,
                    'file_size': os.path.getsize(final_path),
                    **metadata  # Include all extracted metadata
                }

                # Add to database
                db.append(image_data)
                save_database(db)

                flash(f'Image uploaded successfully: {final_filename}', 'success')
                return redirect(url_for('index'))

            except Exception as e:
                # Clean up on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, or WebP files.', 'error')

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
        'special_fields': {}
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