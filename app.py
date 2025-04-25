# app.py
import os
import base64
import requests
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory, stream_with_context, Response, Blueprint
from flask_cors import CORS
from PIL import Image, ExifTags, ImageCms
import xml.etree.ElementTree as ET
import logging
import tempfile

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
logger = logging.getLogger(__name__)

# --- All routes are now under /locaitor via Blueprint for reverse proxy compatibility ---
main = Blueprint(
    'main',
    __name__,
    static_folder='static',
    static_url_path='/static'
)

@main.route('/favicon.ico')
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
    "You are an expert assistant in image forensics, geolocation, and socioeconomic inference. "
    "Given the image and all available metadata (EXIF, HEIC EXIF, XMP, ICC, provenance, AI detection fields), provide a detailed, narrative description of everything you can infer about the photo, its context, and its origin. "
    "Describe the scene, the likely location (landmarks, city, region, address if possible), and any clues about the environment. "
    "If possible, make an educated guess about the direction the photographer was facing (e.g., north, south, toward a landmark, etc.), using shadows, sun position, or metadata. "
    "Critically, analyze the socioeconomic status (SES) of the neighborhood or area where the photo was taken, using all available clues (address, visual cues, metadata, etc.). "
    "If you have access to current web data or APIs, list likely nearby restaurants, amenities, schools, and typical jobs/industries in the area. "
    "If the image appears to be AI-generated or synthetic, or if the metadata suggests this, clearly state so and explain your reasoning. "
    "Always use both visual cues and ALL available metadata. Summarize any provenance or software information you find. "
    "Be as specific, thorough, and accessible as possible in your narrative."
)

def extract_exif(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        exif = img._getexif() or {}
        logger.info(f"EXIF extracted: {list(exif.keys())}")
        return { ExifTags.TAGS.get(k, k): v for k, v in exif.items() }
    except Exception as e:
        logger.error(f"EXIF extraction failed: {e}")
        return {}

def parse_gps(exif):
    """
    Improved GPS extraction from EXIF data to handle various GPS formats.
    Returns (lat, lon) tuple if GPS data is found, otherwise None.
    """
    if not exif:
        return None
        
    # Handle GPSInfo dictionary format
    gps_info = exif.get("GPSInfo")
    if gps_info and isinstance(gps_info, dict):
        gps = { ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items() }
        
        try:
            def to_deg(vals):
                # Handle both simple values and rational values
                if isinstance(vals, tuple) or isinstance(vals, list):
                    d, m, s = vals
                    # Handle rational values stored as (num, denom) tuples
                    if isinstance(d, tuple) and len(d) == 2:
                        d = d[0] / d[1]
                    if isinstance(m, tuple) and len(m) == 2:
                        m = m[0] / m[1]
                    if isinstance(s, tuple) and len(s) == 2:
                        s = s[0] / s[1]
                    return d + m/60 + s/3600
                # Already a decimal value
                elif isinstance(vals, (int, float)):
                    return float(vals)
                return 0
                
            lat = to_deg(gps["GPSLatitude"])
            if gps.get("GPSLatitudeRef", "N") != "N":
                lat = -lat
                
            lon = to_deg(gps["GPSLongitude"])
            if gps.get("GPSLongitudeRef", "E") != "E":
                lon = -lon
                
            logger.info(f"Extracted GPS from GPSInfo: {lat}, {lon}")
            return lat, lon
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Failed to parse GPS from GPSInfo: {e}")
    
    # Handle direct lat/lon fields
    lat_field = None
    lon_field = None
    
    # Try common GPS field names
    gps_field_names = [
        ('GPSLatitude', 'GPSLongitude'),
        ('Latitude', 'Longitude'),
        ('latitude', 'longitude')
    ]
    
    for lat_name, lon_name in gps_field_names:
        if lat_name in exif and lon_name in exif:
            lat_field = lat_name
            lon_field = lon_name
            break
    
    if lat_field and lon_field:
        try:
            lat = float(exif[lat_field])
            lon = float(exif[lon_field])
            
            # Check for reference fields
            lat_ref = exif.get(f"{lat_field}Ref", exif.get("GPSLatitudeRef", "N"))
            lon_ref = exif.get(f"{lon_field}Ref", exif.get("GPSLongitudeRef", "E"))
            
            if lat_ref == "S":
                lat = -lat
            if lon_ref == "W":
                lon = -lon
                
            logger.info(f"Extracted GPS from direct fields: {lat}, {lon}")
            return lat, lon
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse GPS from direct fields: {e}")
    
    # Handle GPSPosition string format
    gps_position = exif.get('GPSPosition')
    if gps_position and isinstance(gps_position, str):
        try:
            # Format: "34.123456 N, 118.123456 W" or "34.123456, -118.123456"
            import re
            coords = re.match(r"([+-]?\d+\.\d+)\s*([NSEWnsew])?[,\s]+([+-]?\d+\.\d+)\s*([NSEWnsew])?", gps_position)
            if coords:
                lat = float(coords.group(1))
                lon = float(coords.group(3))
                
                if coords.group(2) and coords.group(2).upper() == "S":
                    lat = -lat
                if coords.group(4) and coords.group(4).upper() == "W":
                    lon = -lon
                    
                logger.info(f"Extracted GPS from GPSPosition string: {lat}, {lon}")
                return lat, lon
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse GPS from GPSPosition: {e}")
    
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
    Extract provenance info: software, creator tool, ICC desc, C2PA, and additional fields.
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
        # XMP Copyright
        m = re.search(r'<dc:rights[^>]*>(.*?)</dc:rights>', xmp_icc['xmp'])
        if m:
            provenance['copyright_xmp'] = m.group(1)
        m = re.search(r'<dc:creator[^>]*>(.*?)</dc:creator>', xmp_icc['xmp'])
        if m:
            provenance['creator_xmp'] = m.group(1)
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
    # Additional EXIF provenance fields
    extra_tags = [
        'Copyright', 'Artist', 'CameraOwnerName', 'BodySerialNumber', 'LensModel',
        'LensSerialNumber', 'Make', 'Model', 'ExposureTime', 'FNumber', 'ISOSpeedRatings',
        'FocalLength', 'WhiteBalance', 'MeteringMode', 'Flash', 'Orientation', 'ImageDescription',
        'OwnerName', 'SerialNumber', 'LensMake', 'LensSpecification', 'DateTimeOriginal', 'DateTimeDigitized'
    ]
    for tag in extra_tags:
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

@main.route('/')
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

def extract_exif_exiftool(image_bytes: bytes) -> dict:
    """
    Extract EXIF from image bytes using exiftool CLI (fallback for HEIC/HEIF).
    Returns a dict of EXIF tags, or an error dict.
    """
    import subprocess, json, os
    with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(['exiftool', '-j', tmp_path], capture_output=True, text=True, check=True)
        exif = json.loads(result.stdout)[0]
    except Exception as e:
        exif = {'error': f'exiftool failed: {e}'}
    finally:
        os.remove(tmp_path)
    return exif

def extract_exif_heic(image_bytes):
    """
    Try to extract EXIF from HEIC using pillow-heif, pyheif, or exiftool as fallback.
    Returns a dict of EXIF tags, or a user-friendly error if not possible.
    """
    # Try pillow-heif first
    try:
        import pillow_heif
        heif_file = pillow_heif.read_heif(image_bytes)
        if heif_file.info.get('exif'):
            from PIL import Image
            import piexif
            exif_dict = piexif.load(heif_file.info['exif'])
            flat = {}
            for ifd in exif_dict:
                for tag, val in exif_dict[ifd].items():
                    tag_name = piexif.TAGS[ifd][tag]["name"] if tag in piexif.TAGS[ifd] else tag
                    flat[f"{ifd}:{tag_name}"] = val
            logger.info("HEIC EXIF extracted via pillow-heif.")
            return flat
    except Exception as e:
        logger.warning(f"pillow-heif HEIC EXIF extraction failed: {e}")
    # Fallback to pyheif
    try:
        import pyheif
        heif_file = pyheif.read_heif(image_bytes)
        for md in heif_file.metadata or []:
            if md['type'] == 'Exif':
                import piexif
                exif_dict = piexif.load(md['data'])
                flat = {}
                for ifd in exif_dict:
                    for tag, val in exif_dict[ifd].items():
                        tag_name = piexif.TAGS[ifd][tag]["name"] if tag in piexif.TAGS[ifd] else tag
                        flat[f"{ifd}:{tag_name}"] = val
                logger.info("HEIC EXIF extracted via pyheif.")
                return flat
    except Exception as e:
        logger.error(f"pyheif HEIC EXIF extraction failed: {e}")
    # Fallback to exiftool
    logger.info("Trying exiftool fallback for HEIC EXIF extraction.")
    exif = extract_exif_exiftool(image_bytes)
    if 'error' in exif:
        return {"error": f"HEIC EXIF extraction failed due to library incompatibility and exiftool fallback: {exif['error']}"}
    return exif

# Store last analysis for report download
last_analysis_result = None

# --- Utility: DMS to decimal conversion ---
def dms_to_decimal(dms, ref=None):
    """
    Convert EXIF DMS (e.g., '34 deg 25\' 7.43" N') to decimal.
    Accepts string, tuple/list, Fraction, or float.
    """
    import re
    from fractions import Fraction
    
    # Handle Fraction objects
    if isinstance(dms, Fraction):
        return float(dms)
        
    if isinstance(dms, (float, int)):
        return float(dms)
    if isinstance(dms, str):
        match = re.match(r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)[^\d]*([NSEW])?", dms)
        if match:
            deg, min, sec, direction = match.groups()
            dec = float(deg) + float(min)/60 + float(sec)/3600
            if direction in ['S', 'W'] or (ref and ref in ['S', 'W']):
                dec = -dec
            return dec
    if isinstance(dms, (list, tuple)) and len(dms) == 3:
        deg, min, sec = dms
        
        # Handle Fraction objects in tuple elements
        if isinstance(deg, Fraction):
            deg = float(deg)
        if isinstance(min, Fraction):
            min = float(min)
        if isinstance(sec, Fraction):
            sec = float(sec)
            
        dec = float(deg) + float(min)/60 + float(sec)/3600
        if ref in ['S', 'W']:
            dec = -dec
        return dec
    return None

# --- Modular GPS extraction ---
def extract_best_gps(exif_sources):
    """
    exif_sources: list of (name, dict) tuples (heic_exif, exif_orig, exif, parsed_gps)
    Returns: (gps_coords: dict or None, source: str or None)
    Always returns decimal lat/lon if possible.
    """
    # First try pre-parsed GPS coordinates
    for source_name, exif in exif_sources:
        if source_name == 'parsed_gps' and exif and 'lat' in exif and 'lon' in exif:
            # Ensure coordinates are float values
            return {
                'lat': float(exif['lat']), 
                'lon': float(exif['lon'])
            }, source_name
    
    # Try direct GPS fields in all sources
    for source_name, exif in exif_sources:
        if not exif:
            continue
            
        # Process GPS using parse_gps function
        gps_coords = parse_gps(exif)
        if gps_coords and len(gps_coords) == 2:
            lat, lon = gps_coords
            if lat is not None and lon is not None:
                # Ensure coordinates are float values
                return {
                    'lat': float(lat), 
                    'lon': float(lon)
                }, source_name
    
    # If none of the direct GPS extractions worked, try DMS conversion
    for source_name, exif in exif_sources:
        if not exif:
            continue
            
        lat = exif.get('GPSLatitude')
        lon = exif.get('GPSLongitude')
        lat_ref = exif.get('GPSLatitudeRef')
        lon_ref = exif.get('GPSLongitudeRef')
        
        if lat and lon:
            dec_lat = dms_to_decimal(lat, lat_ref)
            dec_lon = dms_to_decimal(lon, lon_ref)
            if dec_lat is not None and dec_lon is not None:
                # Ensure coordinates are float values
                return {
                    'lat': float(dec_lat), 
                    'lon': float(dec_lon)
                }, source_name
    
    return None, None

# --- Modular metadata packaging ---
def package_metadata(exif, exif_orig, heic_exif, xmp_icc, provenance, is_ai, ai_reason):
    return {
        'exif': exif,
        'exif_orig': exif_orig,
        'heic_exif': heic_exif,
        'xmp_icc': xmp_icc,
        'provenance': provenance,
        'ai_generated': is_ai,
        'ai_reason': ai_reason
    }

# --- Modular streaming: metadata first, then LLM output ---
def stream_metadata_and_llm(metadata, gps_coords, address, llm_stream):
    import json
    # 1. Send metadata as JSON line
    yield json.dumps({'metadata': metadata, 'gps': gps_coords, 'address': address}) + '\n'
    # 2. Stream LLM output
    for chunk in llm_stream:
        yield chunk

# Utility: Recursively convert IFDRational and other non-serializable types to float/int/str

def make_json_serializable(obj):
    """
    Recursively convert IFDRational, bytes, Fraction and other non-serializable types in EXIF dicts to float/int/str.
    """
    try:
        from PIL.TiffImagePlugin import IFDRational
    except ImportError:
        IFDRational = None
    
    # Import Fraction for handling fraction objects
    from fractions import Fraction
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif IFDRational and isinstance(obj, IFDRational):
        return float(obj)
    elif 'IFDRational' in str(type(obj)):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    elif isinstance(obj, Fraction):
        return float(obj)
    elif isinstance(obj, bytes):
        # For small blobs, base64 encode; for large, just note it's bytes
        if len(obj) < 128:
            return base64.b64encode(obj).decode('utf-8')
        else:
            return f"<{len(obj)} bytes>"
    else:
        return obj

@main.route('/analyze', methods=['POST'])
def analyze():
    try:
        global last_analysis_result
        if 'image' not in request.files:
            logger.error('No image provided in request.')
            return jsonify({'error': 'No image provided'}), 400

        image = request.files['image']
        raw_orig = image.read()  # Save original bytes
        orig_format = image.mimetype
        heic_exif = {}
        img_format = None
        img = None

        # --- HEIC/Conversion/Resize Handling ---
        heic_error = None
        is_heic = orig_format in ('image/heic', 'image/heif') or (image.filename and image.filename.lower().endswith('.heic'))
        try:
            img = Image.open(BytesIO(raw_orig))
            logger.info(f"Image opened: format={img.format}, mimetype={orig_format}, size={img.size}")
            img_format = img.format if img.format in ('JPEG', 'PNG') else 'JPEG'
        except UnidentifiedImageError as e:
            img = None
            img_format = None
            heic_error = str(e)

        # Always extract HEIC EXIF if the image is HEIC, regardless of how it was opened
        if is_heic:
            heic_exif = extract_exif_heic(raw_orig)
            if 'error' in heic_exif:
                logger.error(f"HEIC EXIF extraction failed: {heic_exif['error']}")

        # If PIL failed to open, try pyheif/pillow-heif for image conversion
        if img is None or img_format is None:
            if pyheif and is_heic:
                try:
                    heif_file = pyheif.read_heif(raw_orig)
                    from PIL import Image as PILImage
                    img = PILImage.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw"
                    )
                    logger.info("HEIC image opened via pyheif.")
                    img_format = 'JPEG'
                except Exception as he:
                    heic_error = str(he)
            elif pillow_heif and is_heic:
                try:
                    heif_file = pillow_heif.read_heif(raw_orig)
                    img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw"
                    )
                    logger.info("HEIC image opened via pillow-heif.")
                    img_format = 'JPEG'
                except Exception as he:
                    heic_error = str(he)

        # Final check before proceeding (outside try/except)
        if img is None or img_format is None:
            logger.error(f'Image could not be opened or format could not be determined. {heic_error or ""}')
            return jsonify({'error': f'Image could not be opened or format could not be determined. {heic_error or ""}'}), 400

        # Extract EXIF and GPS from original bytes (before conversion)
        exif_orig = extract_exif(raw_orig)
        gps_orig = parse_gps(exif_orig)

        # Resize if too large
        if max(img.size) > MAX_DIM:
            scale = MAX_DIM / max(img.size)
            new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
            logger.info(f"Resizing image from {img.size} to {new_size}")
            img = img.resize(new_size, Image.LANCZOS)

        # Save to bytes (JPEG, quality=90)
        buf = BytesIO()
        img.save(buf, format=img_format, quality=90)
        buf.seek(0)
        raw = buf.read()
        if len(raw) > MAX_SIZE:
            logger.error('Image too large after conversion (max 10MiB).')
            return jsonify({'error': 'Image too large after conversion (max 10MiB).'}), 400

        # EXIF, XMP, ICC extraction now uses processed bytes
        exif = extract_exif(raw)
        xmp_icc = extract_xmp_icc(raw)
        provenance = extract_provenance(raw, exif, xmp_icc)
        is_ai, ai_reason = detect_ai_generator(exif, xmp_icc, provenance)
        logger.info(f"AI detection: {is_ai}, Reason: {ai_reason}")

        # --- Make all metadata JSON serializable ---
        exif = make_json_serializable(exif)
        exif_orig = make_json_serializable(exif_orig)
        heic_exif = make_json_serializable(heic_exif)
        xmp_icc = make_json_serializable(xmp_icc)
        provenance = make_json_serializable(provenance)

        # --- Refined fallback: Use heic_exif if exif is empty, else exif_orig ---
        if (not exif or len(exif) == 0):
            if heic_exif and len(heic_exif) > 0:
                exif = heic_exif.copy()
            elif exif_orig and len(exif_orig) > 0:
                exif = exif_orig.copy()

        # --- Compose full metadata for LLM and frontend ---
        all_metadata = package_metadata(exif, exif_orig, heic_exif, xmp_icc, provenance, is_ai, ai_reason)

        # --- Modular GPS extraction: try all sources (heic_exif, exif, exif_orig, parsed_gps) ---
        exif_sources = [
            ('heic_exif', heic_exif),
            ('exif', exif),
            ('exif_orig', exif_orig),
        ]
        # Add parsed GPS as dict if available
        if gps_orig and isinstance(gps_orig, tuple) and len(gps_orig) == 2:
            parsed_gps = {'lat': gps_orig[0], 'lon': gps_orig[1]}
            exif_sources.append(('parsed_gps', parsed_gps))
            
        # Extract GPS coordinates from available sources
        gps_coords, gps_source = extract_best_gps(exif_sources)
        logger.info(f"Final GPS coordinates sent to frontend: {gps_coords} (source: {gps_source})")

        # Compose a detailed metadata string for the LLM, including all fields
        def dict_to_lines(d, prefix=''):
            if not isinstance(d, dict):
                return str(d)
            lines = []
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    lines.extend(dict_to_lines(v, prefix=prefix+'  '))
                else:
                    lines.append(f"{prefix}{k}: {v}")
            return lines

        metadata_lines = []
        metadata_lines.append('EXIF:')
        metadata_lines.extend(dict_to_lines(exif, '  '))
        if heic_exif:
            metadata_lines.append('HEIC_EXIF:')
            metadata_lines.extend(dict_to_lines(heic_exif, '  '))
        metadata_lines.append('XMP/ICC:')
        metadata_lines.extend(dict_to_lines(xmp_icc, '  '))
        metadata_lines.append('Provenance:')
        metadata_lines.extend(dict_to_lines(provenance, '  '))
        metadata_lines.append(f'AI Generated: {is_ai}')
        metadata_lines.append(f'AI Reason: {ai_reason}')
        full_metadata_str = '\n'.join(metadata_lines)

        # Data URL
        data_url = f"data:image/{img_format.lower()};base64,{base64.b64encode(raw).decode()}"

        # Payload (enable streaming)
        payload = {
            "model": MODEL,
            "temperature": 0.0,
            "stream": True,  # Always True for streaming
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": full_metadata_str},
                    {"type": "text", "text": f"Where exactly was this photo taken? Use visual cues and ALL embedded metadata (including EXIF, HEIC EXIF, XMP, ICC, provenance, and AI detection fields). Also, does this image appear to be AI-generated or synthetic? Explain your reasoning. AI detection reason: {ai_reason}. Provenance: {provenance}"},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                  ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type":  "application/json"
        }

        # --- Reverse geocode if possible ---
        address = None
        if gps_coords and gps_coords.get('lat') and gps_coords.get('lon'):
            try:
                url = (
                    f'https://nominatim.openstreetmap.org/reverse?format=json'
                    f'&lat={gps_coords["lat"]}&lon={gps_coords["lon"]}&zoom=18'
                    f'&addressdetails=1&accept-language=en'
                )
                r = requests.get(url, headers={'User-Agent': 'locator-app/1.0'})
                if r.ok:
                    address = r.json().get('display_name')
                    logger.info(f"Reverse geocoded address: {address}")
                else:
                    logger.warning(f"Reverse geocoding failed: {r.text}")
            except Exception as e:
                logger.error(f"Reverse geocoding exception: {e}")

        # --- Streaming xAI response ---
        def llm_stream():
            try:
                with requests.post(XAI_API_URL, headers=headers, json=payload, stream=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            text = chunk.decode(errors='ignore')
                            yield text
            except Exception as e:
                logger.error(f"Streaming xAI API failed: {e}")
                yield f"\n[Error streaming xAI API: {e}]"

        # If client requests streaming, use streaming response
        if request.args.get('stream') == '1':
            try:
                # Include GPS source in the streamed metadata
                return Response(stream_with_context(stream_metadata_and_llm({**all_metadata, 'gps_source': gps_source}, gps_coords, address, llm_stream())), mimetype='text/plain')
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                return Response(f"[Error streaming: {e}]", mimetype='text/plain')

        # Fallback: non-streaming (for legacy clients)
        payload["stream"] = False
        try:
            r = requests.post(XAI_API_URL, headers=headers, json=payload)
            try:
                r.raise_for_status()
                if r.headers.get('content-type', '').startswith('application/json'):
                    data = r.json()
                    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    logger.error(f"xAI API non-JSON response: {r.status_code} {r.text}")
                    return jsonify({"error": f"xAI API returned non-JSON response: {r.status_code} {r.text[:200]}" }), 502
            except Exception as e:
                logger.error(f"xAI API request failed: {e}, response: {r.text[:200]}")
                return jsonify({"error": f"xAI API request failed: {e}, response: {r.text[:200]}" }), 502
            logger.info(f"xAI API response received.")
            result = {
                "content": answer,
                "metadata": {**all_metadata, 'gps_source': gps_source},
                "gps": gps_coords,
                "address": address
            }
            last_analysis_result = result  # Store for report download
            return jsonify(result)
        except requests.RequestException as e:
            logger.error(f"xAI API request failed: {e}")
            return jsonify({"error": str(e)}), 502
    except Exception as e:
        logger.error(f"/analyze endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@main.route('/report', methods=['GET'])
def download_report():
    """
    Download the last analysis as a JSON file.
    """
    from flask import Response
    import json
    if not last_analysis_result:
        return jsonify({'error': 'No analysis available yet.'}), 404
    response = Response(json.dumps(last_analysis_result, indent=2), mimetype='application/json')
    response.headers['Content-Disposition'] = 'attachment; filename=analysis_report.json'
    return response

@main.route('/logs', methods=['GET'])
def download_logs():
    """
    Download the current log file.
    """
    log_path = logging.getLogger().handlers[0].baseFilename if logging.getLogger().handlers else 'app.log'
    if not os.path.exists(log_path):
        return jsonify({'error': 'Log file not found.'}), 404
    return send_from_directory(os.path.dirname(log_path), os.path.basename(log_path), as_attachment=True)

# --- Reverse Geocoding Endpoint ---
@main.route('/reverse_geocode')
def reverse_geocode():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'error': 'Missing lat/lon'}), 400
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1'
    r = requests.get(url, headers={'User-Agent': 'locator-app/1.0'})
    if r.ok:
        return jsonify(r.json())
    return jsonify({'error': 'Reverse geocoding failed'}), 502

# Register the Blueprint with the /locaitor prefix
app.register_blueprint(main, url_prefix='/locaitor')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5002, debug=True)