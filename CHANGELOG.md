# Changelog

## [Unreleased]
- Enhanced backend to extract XMP and ICC profile metadata in addition to EXIF.
- Added AI/synthetic image detection using metadata (EXIF, XMP, ICC) and known AI generator tags.
- Updated LLM prompt to explicitly ask for both location and AI/synthetic status, with reasoning.
- API now returns a `metadata` field with all extracted metadata and AI detection results.
- Updated documentation and usage examples.
- Expanded provenance extraction: now includes copyright, artist, camera/lens serial, lens info, owner, exposure, ISO, and more from EXIF/XMP.
- Added /report endpoint to download the last analysis as JSON.
- Added /logs endpoint to download the current server log file.
- Frontend: Added accessible Download Report and Download Logs buttons below metadata section.
- All new features are accessible (ARIA labels, keyboard, alerts for errors).
- Always display a map: shows world map if no GPS, or location if GPS present.
- Show both original and processed EXIF metadata in the UI for transparency.
- Accessibility: ARIA/alt text for map and metadata, keyboard navigation for map marker.
- Fixed: Consistent GPS extraction: now always uses original image bytes for EXIF/GPS, preventing mismatches between LLM and UI. 