# English Learning Helper

A powerful tool that loads vocabulary words from files, generates native speaker sentences using Google AI, and creates interactive fill-in-the-blank tests.

## Features

- 📖 Load vocabulary words from text files (supports Kindle HTML exports)
- 🤖 Generate natural sentences using Google Generative AI
- 📝 Create fill-in-the-blank tests
- 🎯 Interactive testing with scoring
- 📊 Rich terminal interface with colors and tables
- 🖥️ **NEW: GUI Application** with text editor-style results display
- 🔄 Track processed words across multiple sessions
- ⭐ Mark duplicate words as IMPORTANT for reinforcement

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Google AI API key (REQUIRED for sentence generation):**
   - Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project directory:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   - **Important**: Without the API key, the application will use mock sentence generation which provides basic examples but not AI-generated natural sentences.
   - The API key is free and allows you to generate realistic, conversational example sentences using Google's Gemini AI.

## Usage

### GUI Application (Recommended)

Launch the graphical interface:

```bash
python english_learner_gui.py
```

**Features:**
- 🖱️ Easy file selection with browse dialogs
- 📝 Text editor-style results display with syntax highlighting
- ⚙️ All options accessible through the interface
- 📊 Real-time progress updates
- 💾 Automatic output file management

**GUI Options:**
- **Input File**: Select your vocabulary file (supports HTML, TXT, MD)
- **Output File**: Choose where to save results (auto-timestamped if not specified)
- **Examples per word**: Number of example sentences (default: 1)
- **Words per section**: Group words into sections (default: 20)
- **Test batch size**: Questions per test batch (default: 20)
- **Track new words**: Enable/disable word tracking across sessions
- **Tracker file**: Location of the word tracking cache

### Command Line Interface

#### Process a Vocabulary File

```bash
python english_learner.py process "path/to/your/vocabulary_file.html"
```

**Options:**
- `--output`: Output file path (default: auto-generated with timestamp)
- `--examples-per-word`: Number of example sentences per word (default: 1)
- `--words-per-section`: Words per section in output (default: 20)
- `--test-batch-size`: Questions per test batch (default: 20)
- `--track-new-words`: Track processed words (default: True)
- `--tracker-file`: Path to tracker file (default: processed_words.json)

**Example:**
```bash
python english_learner.py process "kindle_export.html" --output vocab.txt --examples-per-word 1
```

#### Check New Words (Preview)

```bash
python english_learner.py check-new "path/to/file.html"
```

#### Clear Word Tracker

```bash
python english_learner.py clear-tracker
```

#### Generate Sentences for a Specific Word

```bash
python english_learner.py word "innovation" --count 5
```

### File Format

Your vocabulary file should contain words separated by spaces, newlines, or punctuation. The program will automatically extract and clean the vocabulary words.

**Example file content:**
```
innovation disruptive technology
sustainable development
artificial intelligence and machine learning
quantum computing advances
```

## How It Works

1. **Word Extraction**: The program reads your file and extracts vocabulary words, filtering out common stop words and duplicates.

2. **Sentence Generation**: Using Google Generative AI (Gemini), the program creates natural, native-speaker style sentences for each vocabulary word.

3. **Test Creation**: Fill-in-the-blank tests are automatically generated from the sentences, with the target vocabulary words removed.

4. **Interactive Testing**: Run tests to practice and get immediate feedback on your answers.

## Requirements

- Python 3.7+
- Google Generative AI API key
- Internet connection for AI sentence generation

## Dependencies

- `google-generativeai`: For AI-powered sentence generation
- `python-dotenv`: For environment variable management
- `click`: For command-line interface
- `rich`: For beautiful terminal output
- `PyQt5`: For GUI application (Qt framework)
- `beautifulsoup4`: For HTML parsing (Kindle exports)

## Troubleshooting

**API Key Issues**: Make sure your `GOOGLE_API_KEY` is set correctly in the `.env` file.

**File Not Found**: Ensure the file path is correct and the file exists.

**Network Issues**: Sentence generation requires internet access to Google's API.

## License

This project is open source and available under the MIT License.
