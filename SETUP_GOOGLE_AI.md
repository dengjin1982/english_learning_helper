# Setting Up Google AI for Sentence Generation

## Why You Need Google AI

The English Learning Helper uses **Google's Gemini AI** to generate realistic, conversational example sentences that sound like how native US English speakers actually talk. Without the API key, the app uses basic mock sentences which are less natural.

## Quick Setup (3 Steps)

### Step 1: Get Your Free API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key" or "Create API Key"
4. Copy your API key (it will look like: `AIzaSy...`)

### Step 2: Create .env File

Create a file named `.env` in the project directory (`english_learning_helper`) with this content:

```
GOOGLE_API_KEY=your_api_key_here
```

**Replace `your_api_key_here` with your actual API key from Step 1.**

### Step 3: Restart the Application

Close and restart the GUI or command-line application. The warning message should disappear and you'll see:
```
[OK] Using Google AI model: gemini-1.5-flash
```

## Example .env File

```
GOOGLE_API_KEY=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567
```

## Verification

When you run the application, you should see:
- ✅ `[OK] Using Google AI model: gemini-1.5-flash` (or similar)
- ❌ No warning about "No Google API key found"

## Troubleshooting

**Problem**: Still seeing the warning message
- **Solution**: Make sure the `.env` file is in the same directory as `english_learner.py`
- **Solution**: Check that the file is named exactly `.env` (not `.env.txt`)
- **Solution**: Restart the application after creating/editing the `.env` file

**Problem**: "Failed to initialize Google AI"
- **Solution**: Check that your API key is correct (no extra spaces)
- **Solution**: Make sure you have internet connection
- **Solution**: Verify your API key is active at [Google AI Studio](https://makersuite.google.com/app/apikey)

**Problem**: API key not working
- **Solution**: Make sure you copied the entire key (it's usually quite long)
- **Solution**: Check that there are no quotes around the key in the `.env` file
- **Solution**: Try regenerating a new API key

## Cost

**Good news**: Google AI Studio provides free API access with generous limits for personal use. You typically won't be charged for normal usage.

## Optional: Additional API Keys

You can also add these optional keys to your `.env` file for enhanced features:

```
# For more accurate definitions (optional)
OXFORD_APP_ID=your_oxford_app_id
OXFORD_API_KEY=your_oxford_api_key

# For alternative definitions (optional)
WORDS_API_KEY=your_words_api_key
```

These are optional - the app works great with just the Google API key!

