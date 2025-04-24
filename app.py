# app.py
import os
import base64
import requests
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image, ExifTags, ImageCms
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

# xAI (Grok-Vision) settings
XAI_API_KEY = os.getenv('XAI_API_KEY', 'xai-8zAk5VIaL3Vxpu3fO3r2aiWqqeVAZ173X04VK2R1m425uYpWOIOQJM3puq1Q38xJ2sHfbq3mX4PBxJXC')
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL       = os.getenv('XAI_MODEL', 'grok-2-vision')

SYSTEM_PROMPT = (
    "You are an assistant that pinpoints exactly where a photo was taken. "
    "Give the most specific location you can (landmarks, city, region)."
)

def extract_exif(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        exif = img._getexif() or {}
        return { ExifTags.TAGS.get(k, k): v for k, v in exif.items() }
    except:
        return {}

def parse_gps(exif):
    gps_info = exif.get("GPSInfo")
    if not gps_info: return None
    gps = { ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items() }
    def to_deg(vals):
        d, m, s = vals
        return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600
    try:
        lat = to_deg(gps["GPSLatitude"])
        if gps.get("GPSLatitudeRef") != "N": lat = -lat
        lon = to_deg(gps["GPSLongitude"])
        if gps.get("GPSLongitudeRef") != "E": lon = -lon
        return lat, lon
    except:
        return None

def extract_xmp_icc(image_bytes):
    """
    Extract XMP and ICC profile data from image bytes.
    Returns a dict with 'xmp' and 'icc' keys if found.
    """
    result = {}
    try:
        img = Image.open(BytesIO(image_bytes))
        # ICC Profile
        if 'icc_profile' in img.info:
            result['icc'] = 'ICC profile present'
        # XMP (if present)
        xmp_start = image_bytes.find(b'<x:xmpmeta')
        xmp_end = image_bytes.find(b'</x:xmpmeta')
        if xmp_start != -1 and xmp_end != -1:
            xmp_str = image_bytes[xmp_start:xmp_end+12].decode(errors='ignore')
            result['xmp'] = xmp_str
    except Exception as e:
        result['error'] = str(e)
    return result

def extract_provenance(image_bytes, exif, xmp_icc):
    """
    Extract provenance info: software, creator tool, ICC desc, C2PA, etc.
    Returns a dict of provenance fields.
    """
    provenance = {}
    # EXIF Software
    if 'Software' in exif:
        provenance['software'] = exif['Software']
    # XMP CreatorTool
    if 'xmp' in xmp_icc and xmp_icc['xmp']:
        import re
        m = re.search(r'<xmp:CreatorTool>(.*?)</xmp:CreatorTool>', xmp_icc['xmp'])
        if m:
            provenance['creator_tool'] = m.group(1)
        # C2PA/CAI signature
        if 'c2pa.org' in xmp_icc['xmp'].lower() or 'contentauthenticity.org' in xmp_icc['xmp'].lower():
            provenance['c2pa'] = True
    # ICC profile description
    if 'icc' in xmp_icc and xmp_icc['icc']:
        provenance['icc'] = xmp_icc['icc']
    # Thumbnail presence
    if 'JPEGThumbnail' in exif or 'thumbnail' in exif:
        provenance['thumbnail'] = True
    # FileSource, SceneType, ProcessingSoftware, HostComputer
    for tag in ['FileSource', 'SceneType', 'ProcessingSoftware', 'HostComputer']:
        if tag in exif:
            provenance[tag.lower()] = exif[tag]
    return provenance

# Improved AI detection logic
AI_GENERATOR_TAGS = [
    'Stable Diffusion', 'Midjourney', 'DALL-E', 'DreamStudio', 'Craiyon',
    'AI Generated', 'Generative', 'C2PA', 'Firefly', 'Runway', 'Leonardo',
    'DeepFloyd', 'BlueWillow', 'NightCafe', 'NovelAI', 'Diffusion', 'SDXL',
    'AIGC', 'AIGenerated', 'AIGen', 'AIGeneratedContent',
    'contentauthenticity.org', 'c2pa.org'
]
SOFTWARE_SUSPICIOUS = [
    'Stable Diffusion', 'Midjourney', 'DALL-E', 'AI Generated', 'Craiyon',
    'Firefly', 'Runway', 'Leonardo', 'DeepFloyd', 'BlueWillow', 'NightCafe',
    'NovelAI', 'Diffusion', 'SDXL', 'AIGC', 'AIGenerated', 'AIGen', 'AIGeneratedContent',
    'contentauthenticity.org', 'c2pa.org'
]

def detect_ai_generator(exif, xmp_icc, provenance):
    """
    Improved: Only flag as AI-generated if known tags, suspicious software, or C2PA/CAI. If all metadata missing, say 'cannot determine'.
    """
    # Check EXIF fields for known AI tags
    for k, v in exif.items():
        if any(tag.lower() in str(v).lower() for tag in AI_GENERATOR_TAGS):
            return True, f"AI generator tag found in EXIF: {k}: {v}"
    # Check XMP
    if 'xmp' in xmp_icc and xmp_icc['xmp']:
        for tag in AI_GENERATOR_TAGS:
            if tag.lower() in xmp_icc['xmp'].lower():
                return True, f"AI generator tag found in XMP: {tag}"
    # Check ICC
    if 'icc' in xmp_icc and xmp_icc['icc']:
        if any(tag.lower() in xmp_icc['icc'].lower() for tag in AI_GENERATOR_TAGS):
            return True, f"AI generator tag found in ICC: {xmp_icc['icc']}"
    # Check provenance for suspicious software/creator tool
    for key in ['software', 'creator_tool']:
        val = provenance.get(key, '')
        if any(tag.lower() in str(val).lower() for tag in SOFTWARE_SUSPICIOUS):
            return True, f"Suspicious software/creator tool: {val}"
    # C2PA/CAI signature
    if provenance.get('c2pa'):
        return True, "C2PA/Content Authenticity signature found."
    # If all metadata missing, cannot determine
    if not exif and not xmp_icc and not provenance:
        return None, "No metadata found; cannot determine if AI-generated."
    # If no camera model, no software, no date/time, no GPS, no thumbnail, flag as 'possibly synthetic'
    if not exif.get('Model') and not exif.get('Software') and not exif.get('DateTimeOriginal') and not exif.get('GPSInfo') and not provenance.get('thumbnail'):
        return None, "No camera, software, date/time, GPS, or thumbnail metadata; possibly synthetic, but not conclusive."
    return False, "No AI generator tags or suspicious provenance found."

@app.route('/')
def index():
    return render_template('index.html')

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None  # HEIC support not available

try:
    import pyheif
except ImportError:
    pyheif = None  # pyheif not available

from PIL import UnidentifiedImageError

MAX_DIM = 2048
MAX_SIZE = 10 * 1024 * 1024  # 10 MiB

def extract_exif_heic(image_bytes):
    """
    Extract EXIF from HEIC using pyheif if available.
    Returns a dict of EXIF tags, or empty dict if not found.
    """
    if not pyheif:
        return {}
    try:
        heif_file = pyheif.read_heif(image_bytes)
        for md in heif_file.metadata or []:
            if md['type'] == 'Exif':
                import piexif
                exif_dict = piexif.load(md['data'])
                # Flatten exif_dict for easier display
                flat = {}
                for ifd in exif_dict:
                    for tag, val in exif_dict[ifd].items():
                        tag_name = piexif.TAGS[ifd][tag]["name"] if tag in piexif.TAGS[ifd] else tag
                        flat[f"{ifd}:{tag_name}"] = val
                return flat
    except Exception as e:
        return {"error": str(e)}
    return {}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    raw   = image.read()
    orig_format = image.mimetype
    heic_exif = {}

    # --- HEIC/Conversion/Resize Handling ---
    try:
        img = Image.open(BytesIO(raw))
    except UnidentifiedImageError:
        return jsonify({'error': 'Unsupported image format or corrupted file.'}), 400

    # Extract EXIF from HEIC before conversion
    if img.format == 'HEIC' or orig_format in ('image/heic', 'image/heif'):
        if pyheif:
            heic_exif = extract_exif_heic(raw)
        if not pillow_heif:
            return jsonify({'error': 'HEIC support not installed. Please install pillow-heif.'}), 400
        img = img.convert('RGB')
        img_format = 'JPEG'
    else:
        img_format = img.format if img.format in ('JPEG', 'PNG') else 'JPEG'
        if img.mode != 'RGB':
            img = img.convert('RGB')

    # Resize if too large
    if max(img.size) > MAX_DIM:
        scale = MAX_DIM / max(img.size)
        new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Save to bytes (JPEG, quality=90)
    buf = BytesIO()
    img.save(buf, format=img_format, quality=90)
    buf.seek(0)
    raw = buf.read()
    if len(raw) > MAX_SIZE:
        return jsonify({'error': 'Image too large after conversion (max 10MiB).'}), 400

    # EXIF, XMP, ICC extraction now uses processed bytes
    exif = extract_exif(raw)
    xmp_icc = extract_xmp_icc(raw)
    provenance = extract_provenance(raw, exif, xmp_icc)
    is_ai, ai_reason = detect_ai_generator(exif, xmp_icc, provenance)

    # Merge all metadata
    all_metadata = {
        'exif': exif,
        'heic_exif': heic_exif,
        'xmp_icc': xmp_icc,
        'provenance': provenance,
        'ai_generated': is_ai,
        'ai_reason': ai_reason
    }

    # GPS extraction (prefer HEIC EXIF if present)
    gps = None
    if heic_exif:
        gps_keys = [k for k in heic_exif if 'GPS' in k]
        if gps_keys:
            gps = {k: heic_exif[k] for k in gps_keys}
    if not gps:
        gps = parse_gps(exif)
    if isinstance(gps, tuple):
        gps_coords = {'lat': gps[0], 'lon': gps[1]}
    elif isinstance(gps, dict):
        gps_coords = gps
    else:
        gps_coords = None

    parts = []
    if m := exif.get("Model"): parts.append(f"Camera: {m}")
    if dt := exif.get("DateTimeOriginal"): parts.append(f"Taken on: {dt}")
    if gps_coords and isinstance(gps_coords, dict) and 'lat' in gps_coords and 'lon' in gps_coords:
        parts.append(f"GPS: {gps_coords['lat']:.6f}, {gps_coords['lon']:.6f}")
    if 'xmp' in xmp_icc: parts.append("XMP metadata present")
    if 'icc' in xmp_icc: parts.append("ICC profile present")
    if provenance.get('software'): parts.append(f"Software: {provenance['software']}")
    if provenance.get('creator_tool'): parts.append(f"Creator Tool: {provenance['creator_tool']}")
    if provenance.get('c2pa'): parts.append("C2PA/Content Authenticity signature present")
    metadata_str = "Image metadata: " + "; ".join(parts) if parts else "No EXIF/XMP metadata found."

    # Data URL
    data_url = f"data:image/{img_format.lower()};base64,{base64.b64encode(raw).decode()}"

    # Enhanced system prompt
    SYSTEM_PROMPT = (
        "You are an assistant that pinpoints exactly where a photo was taken. "
        "Give the most specific location you can (landmarks, city, region). "
        "If the image appears to be AI-generated or synthetic, or if the metadata suggests this, "
        "clearly state so and explain your reasoning. Always use both visual cues and all available metadata. "
        "Also, summarize any provenance or software information you find."
    )

    # Payload (no max_tokens)
    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
              "role": "user",
              "content": [
                {"type": "text", "text": metadata_str},
                {"type": "text", "text": f"Where exactly was this photo taken? Use visual cues and embedded metadata. Also, does this image appear to be AI-generated or synthetic? Explain your reasoning. AI detection reason: {ai_reason}. Provenance: {provenance}"},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
              ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type":  "application/json"
    }

    try:
        r = requests.post(XAI_API_URL, headers=headers, json=payload)
        r.raise_for_status()
        data   = r.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return jsonify({
            "content": answer,
            "metadata": all_metadata,
            "gps": gps_coords
        })
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5002, debug=True)