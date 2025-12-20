# English Learner - Usage Guide

## Quick Start

```bash
# Process a vocabulary file
python english_learner.py process "vocabulary.html"

# Process multiple files at once
python english_learner.py process file1.html file2.html file3.txt

# Check what new words are in a file (without processing)
python english_learner.py check-new "vocabulary.html"

# Generate sentences for a specific word
python english_learner.py word "innovation" --count 5

# Clear word tracking cache
python english_learner.py clear-tracker
```

---

## Commands

### 1. `process` - Process Vocabulary Files

**Main command** - Processes vocabulary files and generates learning materials.

#### Basic Usage:
```bash
python english_learner.py process "vocabulary.html"
```

#### Process Multiple Files:
```bash
python english_learner.py process file1.html file2.html file3.txt
```

#### Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--examples-per-word` | `1` | Number of example sentences per word |
| `--output` | Auto-generated | Output filename (auto-timestamped) |
| `--test-batch-size` | `20` | Number of questions per test batch |
| `--words-per-section` | `20` | Words per section before test |
| `--track-new-words` | `True` | Track processed words (default: on) |
| `--no-track` | - | Disable word tracking |
| `--tracker-file` | `processed_words.json` | File to track processed words |

#### Examples:

**Basic processing:**
```bash
python english_learner.py process "kindle_export.html"
```

**Custom output and options:**
```bash
python english_learner.py process "vocab.html" \
    --output "my_vocabulary.txt" \
    --examples-per-word 2 \
    --test-batch-size 15 \
    --words-per-section 25
```

**Process multiple files:**
```bash
python english_learner.py process book1.html book2.html notes.txt \
    --output combined_learning.txt
```

**Disable word tracking:**
```bash
python english_learner.py process "vocab.html" --no-track
```

**Custom tracker file:**
```bash
python english_learner.py process "vocab.html" \
    --tracker-file "my_tracker.json"
```

#### What It Does:
1. ✅ Loads vocabulary words from file(s)
2. ✅ Fetches definitions from online dictionaries
3. ✅ Gets pronunciations (IPA notation)
4. ✅ Generates example sentences using Google AI
5. ✅ Creates fill-in-the-blank tests (randomized order)
6. ✅ Groups words into sections
7. ✅ Saves everything to output file with timestamp

#### Output Format:
```
SECTION 1: VOCABULARY WORDS 1-20
1. word: rations /ˈɹæʃənz/
   explanation: A portion of some limited resource...
   example: • During the war, they had to survive on meager rations.

TEST - Section 1, Batch 1 (20 questions)
Question 1: [sentence with blank]
...

ANSWERS:
Question 1: rations
...
```

---

### 2. `check-new` - Preview New Words

**Preview command** - Shows which words are new vs. already processed (without generating materials).

#### Usage:
```bash
python english_learner.py check-new "vocabulary.html"
```

#### Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tracker-file` | `processed_words.json` | File to track processed words |

#### Example:
```bash
python english_learner.py check-new "new_book.html"
```

#### Output:
```
Word Analysis:
  • Total words in file: 150
  • Already processed: 45
  • New words: 105
  • Total tracked: 500

Important words (appeared before):
  1. rations /ˈɹæʃənz/ (appeared 2 time(s))
  2. searing /sɪəɹɪŋ/ (appeared 1 time(s))
  ...

New words found:
  1. innovation /ˌɪnəˈveɪʃən/
  2. sustainable /səˈsteɪnəbəl/
  ...
```

---

### 3. `word` - Generate Sentences for One Word

**Quick test command** - Generates example sentences for a single word.

#### Usage:
```bash
python english_learner.py word "innovation" --count 5
```

#### Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | `3` | Number of sentences to generate |

#### Examples:

**Generate 3 sentences (default):**
```bash
python english_learner.py word "innovation"
```

**Generate 5 sentences:**
```bash
python english_learner.py word "innovation" --count 5
```

#### Output:
```
Sentences for 'innovation':
1. Innovation drives technological progress.
2. Their latest innovation revolutionized the industry.
3. We need more innovation in renewable energy.
```

---

### 4. `clear-tracker` - Reset Word Tracking

**Maintenance command** - Clears the word tracking cache (resets processed words).

#### Usage:
```bash
python english_learner.py clear-tracker
```

#### Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tracker-file` | `processed_words.json` | File to track processed words |
| `--confirm` | `False` | Skip confirmation prompt |

#### Examples:

**Interactive (asks for confirmation):**
```bash
python english_learner.py clear-tracker
```

**Skip confirmation:**
```bash
python english_learner.py clear-tracker --confirm
```

**Custom tracker file:**
```bash
python english_learner.py clear-tracker --tracker-file "my_tracker.json"
```

---

## File Formats Supported

- **HTML files** (`.html`, `.htm`) - Including Kindle note exports
- **Text files** (`.txt`)
- **Markdown files** (`.md`)

The program automatically detects file type and extracts vocabulary words.

---

## Features

### ✅ Multiple File Processing
Process multiple vocabulary files in one command:
```bash
python english_learner.py process file1.html file2.html file3.txt
```

### ✅ Word Tracking
- Tracks processed words across sessions
- Identifies new words vs. duplicates
- Marks duplicate words as "IMPORTANT" for reinforcement
- Generates different examples for duplicate words

### ✅ Parallel Processing
- Uses 20 parallel threads for fast processing
- Processes words simultaneously
- ~3-5x faster than sequential processing

### ✅ Randomized Tests
- Test questions are shuffled (not in same order as explanations)
- More challenging and effective for learning
- Answers still match correctly

### ✅ Pronunciation Support
- Fetches IPA pronunciations automatically
- Uses base word for pronunciation (e.g., "rations" → "ration")
- Displays pronunciation after each word

### ✅ Dictionary Definitions
- Fetches from Free Dictionary API (fastest)
- Falls back to Oxford Dictionary API (most accurate)
- Uses original word for definitions

---

## Environment Variables

Create a `.env` file in the project directory:

```env
# Required for AI sentence generation
GOOGLE_API_KEY=your_google_api_key_here

# Optional: For more accurate definitions
OXFORD_APP_ID=your_oxford_app_id
OXFORD_API_KEY=your_oxford_api_key

# Optional: Alternative definitions source
WORDS_API_KEY=your_words_api_key
```

Get Google API key: https://makersuite.google.com/app/apikey

---

## Common Workflows

### Workflow 1: First Time Processing
```bash
# Process your first vocabulary file
python english_learner.py process "book1.html"

# Output: vocabulary_learning_materials_20251220_120000.txt
```

### Workflow 2: Processing New Material
```bash
# Check what's new
python english_learner.py check-new "book2.html"

# Process only new words
python english_learner.py process "book2.html"
```

### Workflow 3: Combine Multiple Sources
```bash
# Process multiple books/articles at once
python english_learner.py process book1.html book2.html article.txt \
    --output combined_learning.txt
```

### Workflow 4: Custom Configuration
```bash
# More examples, larger sections, smaller test batches
python english_learner.py process "vocab.html" \
    --examples-per-word 3 \
    --words-per-section 30 \
    --test-batch-size 10
```

---

## Output File Structure

```
VOCABULARY LEARNING MATERIALS
================================================================================
Total Vocabulary Words: 150

SECTION 1: VOCABULARY WORDS 1-20
================================================================================
1. word: rations /ˈɹæʃənz/
--------------------------------------------------------------------------------
explanation:
A portion of some limited resource allocated to a person or group.

example:
  • During the war, they had to survive on meager rations.

[More words...]

TEST - Section 1, Batch 1 (20 questions)
--------------------------------------------------------------------------------
Question 1:
[sentence with blank]

ANSWERS:
--------------------------------------------------------------------------------
Question 1: rations
  Full sentence: [complete sentence]

[Next section...]
```

---

## Tips

1. **Use word tracking**: Keep it enabled to avoid reprocessing words
2. **Process in batches**: Process related files together for better organization
3. **Check before processing**: Use `check-new` to preview what will be processed
4. **Custom output names**: Use `--output` to organize your learning materials
5. **Multiple examples**: Use `--examples-per-word 2` for more examples

---

## Troubleshooting

**Problem**: "No Google API key found"
- **Solution**: Create `.env` file with `GOOGLE_API_KEY=your_key`
- **See**: `SETUP_GOOGLE_AI.md` for detailed instructions

**Problem**: File not found
- **Solution**: Check file path (can use relative or absolute paths)
- **Solution**: Program auto-detects extensions (.html, .txt, .md)

**Problem**: Too slow
- **Solution**: Already optimized with parallel processing (20 threads)
- **Solution**: Check internet connection speed
- **Solution**: Processing is I/O-bound (waiting for API responses)

**Problem**: Rate limit errors
- **Solution**: Reduce parallel workers in code (currently 20)
- **Solution**: Add small delays between API calls

---

## GUI Alternative

For a graphical interface, use:
```bash
python english_learner_gui.py
```

Or double-click: `launch_gui.bat` (Windows)

The GUI supports all the same features with a user-friendly interface.

