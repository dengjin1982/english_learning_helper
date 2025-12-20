# Multiple Files Support

## Overview

The English Learning Helper now supports processing **multiple vocabulary files in one shot**! This allows you to combine words from multiple sources and process them all together.

## Features

### ✅ CLI Support
Process multiple files from command line:
```bash
python english_learner.py process file1.html file2.html file3.txt
```

### ✅ GUI Support
- Select multiple files at once using the file browser
- Hold **Ctrl** (Windows/Linux) or **Cmd** (Mac) to select multiple files
- Files are displayed separated by semicolons
- All files are processed together and combined

## How It Works

1. **File Selection**: Select one or more vocabulary files
2. **Word Extraction**: Words are extracted from all files
3. **Deduplication**: Duplicate words are automatically removed (preserves order)
4. **Processing**: All unique words are processed together
5. **Output**: Single output file with all words combined

## Usage Examples

### Command Line

**Single file (as before):**
```bash
python english_learner.py process "vocabulary.html"
```

**Multiple files:**
```bash
python english_learner.py process "book1.html" "book2.html" "notes.txt"
```

**With options:**
```bash
python english_learner.py process file1.html file2.html --output combined.txt --examples-per-word 2
```

### GUI

1. Click **"Browse..."** button next to "Input File"
2. In the file dialog, hold **Ctrl** (or **Cmd** on Mac) and click multiple files
3. Selected files will appear in the input field, separated by semicolons
4. Click **"Process File"** to process all files together

## Benefits

- **Combine multiple sources**: Merge vocabulary from different books, articles, or notes
- **Single output**: Get one comprehensive learning material file
- **Automatic deduplication**: No duplicate words in the output
- **Efficient processing**: All files processed in parallel
- **Word tracking**: Tracks words across all files together

## File Format Support

All supported formats work with multiple files:
- HTML files (`.html`, `.htm`) - including Kindle exports
- Text files (`.txt`)
- Markdown files (`.md`)

## Example Workflow

1. You have vocabulary from 3 different books:
   - `elon_musk_book.html`
   - `steve_jobs_book.html`
   - `personal_notes.txt`

2. Process them all at once:
   ```bash
   python english_learner.py process elon_musk_book.html steve_jobs_book.html personal_notes.txt
   ```

3. Result: One output file with all unique words combined, ready for learning!

## Technical Details

- Files are processed sequentially for loading
- Words are combined and deduplicated
- Processing happens in parallel (20 threads)
- Word tracking works across all files
- Output format remains the same (sections, tests, answers)

## Notes

- **File paths**: Can use relative or absolute paths
- **Extensions**: Auto-detects common extensions if not specified
- **Errors**: If any file is not found, processing stops with an error message
- **Performance**: Processing multiple files is just as fast as single files (parallel processing)

## Troubleshooting

**Problem**: Files not found
- **Solution**: Check file paths are correct
- **Solution**: Try using absolute paths

**Problem**: Too many files selected
- **Solution**: Process in smaller batches if needed
- **Solution**: No hard limit, but very large batches may take longer

**Problem**: Duplicate words across files
- **Solution**: This is handled automatically - duplicates are removed
- **Solution**: Each unique word appears only once in the output

