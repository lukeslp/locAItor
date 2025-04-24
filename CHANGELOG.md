# Changelog

## [Unreleased]
- Enhanced backend to extract XMP and ICC profile metadata in addition to EXIF.
- Added AI/synthetic image detection using metadata (EXIF, XMP, ICC) and known AI generator tags.
- Updated LLM prompt to explicitly ask for both location and AI/synthetic status, with reasoning.
- API now returns a `metadata` field with all extracted metadata and AI detection results.
- Updated documentation and usage examples. 