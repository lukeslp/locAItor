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