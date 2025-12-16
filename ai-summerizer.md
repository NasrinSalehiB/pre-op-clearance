
## How to use ai summarizer
### Updates
Local model support (default, recommended):
Uses facebook/bart-large-cnn via transformers
Free, no internet after first download
No API keys required
Works offline
Hugging Face API (online):
Tries multiple free models
Falls back if one fails
Google Gemini API:
Free tier support
Requires API key
Dynamic summary generation:
Removed fixed mock summaries
Generates summaries from the prompt using the selected AI

```bash
from ai_summarizer import summarize_risk_assessment

# Option 1: Local model (recommended - free, no internet after first run)
result = summarize_risk_assessment(risk_assessment, api_provider="local")

# Option 2: Hugging Face API (free, requires internet)
result = summarize_risk_assessment(risk_assessment, api_provider="huggingface")

# Option 3: Google Gemini (free tier, requires API key)
result = summarize_risk_assessment(
    risk_assessment, 
    api_provider="gemini",
    api_key="your-key"
)
```
The local model downloads once (~1.6GB) and then works offline. Summaries are generated dynamically from your risk assessment data, not from fixed templates.
The code is ready to use. Run python ai_summarizer.py to test it with the sample input.