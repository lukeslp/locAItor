# 📍 Where Was This Photo Taken? (Demo)

A Flask-based demo app to pinpoint where a photo was taken using both visual cues and comprehensive image metadata. The app also detects and flags AI-generated or synthetic images using metadata and visual analysis.

---

## Features

- **Location Detection:** Uses xAI Grok-Vision to identify the most specific location possible (landmarks, city, region).
- **Comprehensive Metadata Extraction:** Extracts EXIF, XMP, and ICC profile data for every uploaded image.
- **AI/Synthetic Image Detection:**
  - Scans metadata for known AI generator tags (e.g., Stable Diffusion, DALL-E, Midjourney, C2PA, etc.).
  - Flags images with missing or suspicious metadata (e.g., no camera model).
  - LLM prompt asks for both location and AI/synthetic status, with reasoning.
- **Accessible API:** Returns both the LLM's answer and all extracted metadata for transparency.
- **Frontend:** Simple, accessible drag-and-drop UI (see `templates/index.html`).
- **Always displays a map:** If GPS is present, shows the location; if not, shows a world map centered at (0,0).
- **EXIF Consistency:** Both original and processed EXIF metadata are extracted and displayed for transparency.
- **Accessibility:** Map and metadata sections include ARIA/alt text and keyboard navigation support.

---

## Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install flask flask-cors pillow requests pillow-heif pyheif piexif
   ```
2. **Set API credentials:**
   - The app uses environment variables for the xAI API key and model, but defaults are provided for demo purposes.
3. **Run the app:**
   ```bash
   python app.py
   ```
4. **Open in browser:**
   - Go to [http://127.0.0.1:5002](http://127.0.0.1:5002)

---

## API Endpoint

### `POST /analyze`

- **Request:**
  - `multipart/form-data` with an `image` file field.
- **Response:**
  - `content`: LLM's answer (location + AI/synthetic status)
  - `metadata`: Dictionary containing:
    - `exif`: Extracted EXIF metadata (JPEG/PNG)
    - `heic_exif`: Extracted EXIF metadata from HEIC (if present)
    - `xmp_icc`: Extracted XMP and ICC profile data
    - `ai_generated`: Boolean, whether AI generator was detected
    - `ai_reason`: Reason for AI/synthetic flag
  - `gps`: GPS coordinates (if found), as `{ "lat": float, "lon": float }`

#### Example Response
```json
{
  "content": "This photo was taken at the Eiffel Tower, Paris. The image appears to be real, as the metadata includes a camera model and no AI generator tags are present.",
  "metadata": {
    "exif": {"Model": "Canon EOS 80D", ...},
    "heic_exif": {"Model": "Canon EOS 80D", ...},
    "xmp_icc": {"icc": "ICC profile present"},
    "ai_generated": false,
    "ai_reason": "No AI generator tags found.",
    "gps": {"lat": 48.8584, "lon": 2.2945}
  }
}
```

### `GET /report`
- Returns the last analysis as a downloadable JSON file.
- If no analysis has been performed, returns a 404 with an error message.

### `GET /logs`
- Returns the current server log file for download.
- If the log file is missing, returns a 404 with an error message.

---

## Accessibility Considerations

- All API responses are structured for easy parsing and screen reader compatibility.
- The frontend (see `templates/index.html`) uses semantic HTML, ARIA roles, and high-contrast color schemes.
- Error messages are clear and accessible.
- All new buttons are keyboard accessible and have ARIA labels.
- Download actions provide alerts if unavailable.
- No visual-only cues; all actions are announced or visible in text.

---

## References & Inspiration
- [image-discriminator-2 (GitHub)](https://github.com/westonslayton/image-discriminator-2)
- [AI-Generated-Images-vs-Real-Images (GitHub)](https://github.com/roydendsouza31/AI-Generated-Images-vs-Real-Images)
- [awesome-ai-agents (GitHub)](https://github.com/e2b-dev/awesome-ai-agents)

---

## Changelog
See `CHANGELOG.md` for updates.

## Image Format Support & Processing

- **Supported formats:** JPEG, PNG, HEIC/HEIF (with `pillow-heif` installed)
- **Automatic conversion:** HEIC/HEIF images are converted to JPEG for processing and API submission.
- **EXIF from HEIC:** If `pyheif` and `piexif` are installed, EXIF is extracted from HEIC before conversion for maximum metadata fidelity.
- **Resizing:** Images are resized to a maximum dimension of 2048px if needed, and compressed to stay under 10MiB (xAI API limit).
- **Error handling:** Unsupported or oversized images are rejected with a clear error message.

**To enable full HEIC/HEIF support:**
```bash
pip install pillow-heif pyheif piexif
```

---

## CLI Test Suite (Planned)

A CLI test script (`cli_test.py`) will allow you to:
- Upload images (JPEG, PNG, HEIC) to the `/analyze` endpoint
- Validate metadata extraction, AI detection, and API response
- Test image resizing/conversion pipeline
- Run batch tests for regression and automation 

## New Features (Unreleased)

- **Expanded Provenance Fields:**
  - The backend now extracts additional provenance and metadata fields from EXIF and XMP, including copyright, artist, camera/lens serial, lens info, owner, exposure, ISO, and more. These are shown in the frontend under Provenance.
- **Downloadable Reports and Logs:**
  - After analyzing an image, you can now download a JSON report of the last analysis or the server log file directly from the frontend.
  - Endpoints:
    - `GET /report` — Download the last analysis as a JSON file.
    - `GET /logs` — Download the current server log file.
- **Frontend Buttons:**
  - "Download Report" and "Download Logs" buttons are available below the metadata section after analysis.
  - Buttons are accessible (keyboard/ARIA labels) and work with screen readers.

## Usage
1. Upload or drag-and-drop an image.
2. The app analyzes the image, extracts all metadata, and queries the LLM for a location and AI detection.
3. The UI displays:
   - LLM answer
   - AI detection summary
   - Camera, date, GPS, software, and provenance
   - Collapsible panels for EXIF (original and processed), HEIC EXIF, and XMP/ICC
   - A map (always visible)
   - Downloadable report and logs

## Accessibility
- All interactive elements have ARIA labels.
- Map marker is keyboard accessible and has alt text.
- Metadata and results are announced via `aria-live`.

## Setup
- Requires Python 3.8+, Flask, Pillow, requests, and (optionally) pillow-heif/pyheif for HEIC support.
- See `requirements.txt` for dependencies.

## CORS
- CORS is enabled for all endpoints for easy integration.

## Changelog
- See `CHANGELOG.md` for details.

---

For questions or accessibility feedback, contact Luke Steuber. 