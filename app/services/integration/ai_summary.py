"""
AI Summarizer for Pre-Operative Risk Assessment

This module provides functionality to generate concise, clinically-focused summaries
of pre-operative risk assessment reports using AI/LLM models.

The summarizer takes the comprehensive JSON output from the risk assessment system
and generates:
1. Executive summary of key risk factors
2. Clinical narrative for documentation
3. Patient-specific recommendations summary
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import json
import requests
import os

# Try to import transformers for local model support
try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def create_summary_prompt(risk_assessment: Dict[str, Any]) -> str:
    """
    Create a prompt for AI summarization of the risk assessment.

    Args:
        risk_assessment: The full JSON output from integrate_cardiac_and_pulmonary_risk()

    Returns:
        Formatted prompt string for the AI model
    """
    subject_id = risk_assessment.get("subject_id", "Unknown")
    hadm_id = risk_assessment.get("hadm_id", "Unknown")
    surgery_name = risk_assessment.get("surgery", {}).get("name", "Unknown procedure")
    surgery_type = risk_assessment.get("surgery", {}).get("type", "Unknown type")

    # Extract key risk factors
    calculated_risk_factors = risk_assessment.get("calculated_risk_factors", {})
    cardiac_risks = calculated_risk_factors.get("cardiac", {})
    pulmonary_risks = calculated_risk_factors.get("pulmonary", {})

    # Extract RCRI information
    rcri = cardiac_risks.get("RCRI", {})
    rcri_score = rcri.get("score", "N/A")
    rcri_tier = rcri.get("risk_tier", "N/A")
    rcri_percentage = rcri.get("score_percentage", "N/A")

    # Extract ARISCAT information
    ariscat = pulmonary_risks.get("ARISCAT", {})
    ariscat_score = ariscat.get("score", "N/A")
    ariscat_percentage = ariscat.get("score_percentage", "N/A")
    ariscat_tier = pulmonary_risks.get("risk_tier", "N/A")

    # Extract lab summary
    lab_summary = risk_assessment.get("lab_summary", {})

    # Extract recommendations
    recommendations = risk_assessment.get("recommendations", [])

    prompt = f"""You are a clinical assistant helping to summarize a pre-operative risk assessment report.

PATIENT INFORMATION:
- Subject ID: {subject_id}
- Admission ID: {hadm_id}
- Planned Procedure: {surgery_name} ({surgery_type})

CARDIAC RISK ASSESSMENT:
- RCRI Score: {rcri_score} (Risk Tier: {rcri_tier}, Estimated Risk: {rcri_percentage}%)
- RCRI Components: {json.dumps(rcri.get('factors', {}), indent=2)}

PULMONARY RISK ASSESSMENT:
- ARISCAT Score: {ariscat_score} (Estimated Risk: {ariscat_percentage}%, Risk Tier: {ariscat_tier})

KEY LABORATORY VALUES:
{_format_lab_summary_for_prompt(lab_summary)}

CLINICAL RECOMMENDATIONS:
{chr(10).join(f"- {rec}" for rec in recommendations[:5])}

Please provide a concise clinical summary (4-5 sentences) that:
1. Highlights the most critical risk factors
2. Summarizes the overall risk profile (cardiac and pulmonary)
3. Emphasizes key abnormal lab values and their clinical significance
4. Provides a clear recommendation for proceeding with surgery

Write in a professional, clinical tone suitable for medical documentation.
"""

    return prompt


def _format_lab_summary_for_prompt(lab_summary: Dict[str, Any]) -> str:
    """Format lab summary for inclusion in prompt."""
    if not lab_summary:
        return "No recent laboratory values available."

    lines = []
    for lab_key, lab_data in lab_summary.items():
        name = lab_data.get("name", lab_key)
        value = lab_data.get("value")
        normal_range = lab_data.get("normal_range", {})
        captured_ago = lab_data.get("captured_ago", "Unknown")

        if value is not None:
            min_val = normal_range.get("min")
            max_val = normal_range.get("max")
            range_str = (
                f" (Normal: {min_val}-{max_val})"
                if min_val is not None and max_val is not None
                else ""
            )
            status = ""
            if min_val is not None and max_val is not None:
                if value < min_val:
                    status = " [LOW]"
                elif value > max_val:
                    status = " [HIGH]"

            lines.append(
                f"- {name}: {value}{range_str}{status} (Captured: {captured_ago})"
            )

    return "\n".join(lines) if lines else "No recent laboratory values available."


def _call_huggingface_api(
    prompt: str, api_key: Optional[str] = None, model_name: Optional[str] = None
) -> str:
    """
    Call Hugging Face Inference API (free tier available).

    Args:
        prompt: The prompt to send to the model
        api_key: Hugging Face API token (optional for public models)
        model_name: Hugging Face model name (defaults to a reliable free model)

    Returns:
        Generated summary text
    """
    # Use a reliable, commonly available free model
    if not model_name:
        model_name = (
            "microsoft/DialoGPT-medium"  # Fallback, but we'll try better ones first
        )

    # Try multiple free models in order of preference
    models_to_try = [
        "google/flan-t5-large",  # Good for summarization, free
        "facebook/bart-large-cnn",  # Excellent for summarization
        "t5-base",  # General purpose
        "microsoft/DialoGPT-medium",  # Fallback
    ]

    if model_name and model_name not in models_to_try:
        models_to_try.insert(0, model_name)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Format prompt for summarization models
    formatted_prompt = f"""Summarize the following pre-operative risk assessment in 2-3 paragraphs:

{prompt}

Summary:"""

    for model in models_to_try:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"

            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_length": 500,
                    "min_length": 100,
                    "temperature": 0.3,
                    "do_sample": True,
                },
            }

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            # If model is loading, wait a bit
            if response.status_code == 503:
                # Model is loading, wait and retry once
                import time

                time.sleep(10)
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )

            response.raise_for_status()
            result = response.json()

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    text = result[0]["generated_text"].strip()
                    if text:
                        return text
                elif "summary_text" in result[0]:
                    return result[0]["summary_text"].strip()

            # Try direct text extraction
            if isinstance(result, dict):
                if "generated_text" in result:
                    text = result["generated_text"].strip()
                    if text:
                        return text
                elif "summary_text" in result:
                    return result["summary_text"].strip()

            # If we got a response but couldn't parse it, try next model
            if isinstance(result, list) or isinstance(result, dict):
                continue

        except requests.exceptions.RequestException:
            # Try next model
            continue
        except Exception:
            # Try next model
            continue

    # If all models failed, return error message
    return "Unable to connect to free AI services. Please check your internet connection or try using mock mode (use_mock=True)."


def _call_google_gemini_api(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call Google Gemini API (free tier available).

    Args:
        prompt: The prompt to send to the model
        api_key: Google AI API key

    Returns:
        Generated summary text
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            return "Error: Google AI API key not provided. Set GOOGLE_AI_API_KEY environment variable or pass api_key parameter."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are a clinical assistant helping to summarize pre-operative risk assessments.\n\n{prompt}\n\nProvide a concise clinical summary (2-3 paragraphs) in a professional, clinical tone suitable for medical documentation."
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 500,
        },
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()

        return f"Unexpected response format: {result}"

    except requests.exceptions.RequestException as e:
        return f"Error calling Google Gemini API: {str(e)}. Please check your internet connection and API key."


def _call_local_model(prompt: str) -> str:
    """
    Use a local transformer model for summarization (completely free, no API needed).

    Args:
        prompt: The prompt to summarize

    Returns:
        Generated summary text
    """
    if not TRANSFORMERS_AVAILABLE:
        return "Error: transformers library not installed. Install with: pip install transformers torch"

    try:
        # Use a lightweight summarization model
        summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=-1  # Use CPU
        )

        # Format the prompt for summarization
        text_to_summarize = f"""Pre-operative Risk Assessment Summary Request:

{prompt}

Please provide a concise clinical summary."""

        # Truncate if too long (BART has token limits)
        max_length = 1024
        if len(text_to_summarize) > max_length:
            text_to_summarize = text_to_summarize[:max_length]

        result = summarizer(
            text_to_summarize, max_length=500, min_length=100, do_sample=False
        )

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("summary_text", "").strip()
        elif isinstance(result, dict):
            return result.get("summary_text", "").strip()

        return str(result)

    except Exception as e:
        return f"Error using local model: {str(e)}. Try using api_provider='huggingface' for online API."


def summarize_risk_assessment(
    risk_assessment: Dict[str, Any],
    api_provider: str = "local",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    use_mock: bool = False,
) -> Dict[str, Any]:
    """
    Generate an AI-powered summary of the risk assessment using a free AI agent.

    Args:
        risk_assessment: The full JSON output from integrate_cardiac_and_pulmonary_risk()
        api_provider: Which free AI to use:
            - "local" (default): Uses local transformers model (completely free, no internet)
            - "huggingface": Uses Hugging Face Inference API (free, requires internet)
            - "gemini": Uses Google Gemini API (free tier, requires API key)
        api_key: API key/token for the service (optional for Hugging Face public models)
        model_name: Name of the model to use (optional, uses defaults if not provided)
        use_mock: If True, return a mock summary instead of calling the AI (default: False)

    Returns:
        Dictionary with:
        - summary: The generated clinical summary
        - prompt: The prompt used for generation
        - metadata: Information about the summarization

    Examples:
        # Using local model (completely free, no internet needed)
        result = summarize_risk_assessment(risk_assessment, api_provider="local")

        # Using Hugging Face API (free, requires internet)
        result = summarize_risk_assessment(risk_assessment, api_provider="huggingface")

        # Using Google Gemini (free tier, requires API key)
        result = summarize_risk_assessment(
            risk_assessment,
            api_provider="gemini",
            api_key="your-google-ai-api-key"
        )
    """
    prompt = create_summary_prompt(risk_assessment)

    if use_mock:
        # Return a mock summary for demonstration
        summary = _generate_mock_summary(risk_assessment)
        model_used = "mock"
    else:
        # Call the actual free AI
        if api_provider.lower() == "local":
            summary = _call_local_model(prompt)
            model_used = "facebook/bart-large-cnn (local)"

        elif api_provider.lower() == "huggingface":
            # Try to get API key from environment if not provided
            if not api_key:
                api_key = os.getenv("HUGGINGFACE_API_KEY")

            summary = _call_huggingface_api(prompt, api_key, model_name)
            model_used = model_name or "huggingface-api"

        elif api_provider.lower() == "gemini":
            if not api_key:
                api_key = os.getenv("GOOGLE_AI_API_KEY")
            summary = _call_google_gemini_api(prompt, api_key)
            model_used = "gemini-pro"

        else:
            summary = f"Unknown API provider: {api_provider}. Use 'local', 'huggingface', or 'gemini'."
            model_used = "unknown"

    return {
        "summary": summary,
        "prompt": prompt,
        "metadata": {
            "model_used": model_used,
            "api_provider": api_provider if not use_mock else "mock",
            "subject_id": risk_assessment.get("subject_id"),
            "hadm_id": risk_assessment.get("hadm_id"),
        },
    }


def _generate_mock_summary(risk_assessment: Dict[str, Any]) -> str:
    """Generate a mock clinical summary for demonstration purposes."""
    subject_id = risk_assessment.get("subject_id", "Unknown")
    surgery_name = risk_assessment.get("surgery", {}).get("name", "Unknown procedure")

    calculated_risk_factors = risk_assessment.get("calculated_risk_factors", {})
    cardiac_risks = calculated_risk_factors.get("cardiac", {})
    pulmonary_risks = calculated_risk_factors.get("pulmonary", {})

    rcri = cardiac_risks.get("RCRI", {})
    rcri_tier = rcri.get("risk_tier", "unknown")
    rcri_percentage = rcri.get("score_percentage", "N/A")

    ariscat_tier = pulmonary_risks.get("risk_tier", "unknown")
    ariscat_percentage = pulmonary_risks.get("ARISCAT", {}).get(
        "score_percentage", "N/A"
    )

    recommendations = risk_assessment.get("recommendations", [])
    top_recommendation = (
        recommendations[0]
        if recommendations
        else "Standard pre-operative monitoring recommended."
    )

    summary = f"""PRE-OPERATIVE RISK ASSESSMENT SUMMARY - Subject {subject_id}

This patient presents for {surgery_name} with a {rcri_tier} cardiac risk profile (estimated risk: {rcri_percentage}%) 
and {ariscat_tier} pulmonary risk (estimated risk: {ariscat_percentage}%). 

The primary concern is {top_recommendation.lower()}

Key risk factors include: {_extract_key_risk_factors(risk_assessment)}

Recommendation: Proceed with appropriate monitoring and risk mitigation strategies as outlined in the detailed assessment.
"""

    return summary


def _extract_key_risk_factors(risk_assessment: Dict[str, Any]) -> str:
    """Extract key risk factors for summary."""
    factors = []

    calculated_risk_factors = risk_assessment.get("calculated_risk_factors", {})
    cardiac_risks = calculated_risk_factors.get("cardiac", {})

    # Check RCRI factors
    rcri_factors = cardiac_risks.get("RCRI", {}).get("factors", {})
    if rcri_factors.get("Ischemic_heart_disease"):
        factors.append("ischemic heart disease")
    if rcri_factors.get("CHF_history"):
        factors.append("heart failure history")
    if rcri_factors.get("Cerebrovascular_disease"):
        factors.append("cerebrovascular disease")
    if rcri_factors.get("High_risk_surgery"):
        factors.append("high-risk surgery")

    # Check lab abnormalities
    lab_summary = risk_assessment.get("lab_summary", {})
    if lab_summary.get("bnp", {}).get("value"):
        bnp_val = lab_summary["bnp"]["value"]
        if bnp_val > 300:
            factors.append(f"elevated BNP ({bnp_val:.0f} pg/mL)")

    if lab_summary.get("creatinine", {}).get("value"):
        creat_val = lab_summary["creatinine"]["value"]
        creat_max = lab_summary["creatinine"].get("normal_range", {}).get("max", 1.3)
        if creat_val > creat_max:
            factors.append(f"elevated creatinine ({creat_val:.2f} mg/dL)")

    return ", ".join(factors) if factors else "standard risk factors"


# Example usage and sample input
if __name__ == "__main__":
    # Sample input - this would come from integrate_cardiac_and_pulmonary_risk()
    sample_input = {
        "subject_id": 249,
        "hadm_id": 116935,
        "surgery_time": "2149-12-20T10:00:00",
        "last_updated": "2024-01-15T14:30:00",
        "calculated_risk_factors": {
            "cardiac": {
                "RCRI": {
                    "score": 2,
                    "risk_tier": "moderate",
                    "score_percentage": 6.6,
                    "component_fraction": "2/6",
                    "factors": {
                        "High_risk_surgery": True,
                        "Ischemic_heart_disease": True,
                        "CHF_history": False,
                        "Insuline_therapy_for_DM": None,
                        "Cerebrovascular_disease": False,
                        "Preop_creatine>2_mg/dl": False,
                    },
                },
                "AUB_HAS2": {"score": 3, "risk_tier": "high", "score_percentage": 17.0},
                "NSQIP_MACE": {
                    "score": 0.15,
                    "risk_tier": "moderate",
                    "score_percentage": 15.0,
                },
                "Gupta_MICA": {
                    "score": 0.12,
                    "risk_tier": "moderate",
                    "score_percentage": 12.0,
                },
            },
            "pulmonary": {
                "ARISCAT": {
                    "score": 22,
                    "score_percentage": 1.6,
                    "contributors": [
                        "Age 74 years: 3 points",
                        "SpO2 95%: 8 points",
                        "Respiratory infection: 7 points",
                        "Other factors: 4 points",
                    ],
                },
                "risk_tier": "low",
                "COPD_risk_tier": "moderate",
            },
        },
        "lab_summary": {
            "hgb": {
                "name": "Hemoglobin",
                "value": 11.7,
                "normal_range": {"min": 12.0, "max": 15.0},
                "captured_ago": "2 days ago",
                "capture_time": "2149-12-18T06:35:00",
            },
            "creatinine": {
                "name": "Creatinine",
                "value": 1.7,
                "normal_range": {"min": 0.6, "max": 1.1},
                "captured_ago": "1 day ago",
                "capture_time": "2149-12-19T18:40:00",
            },
            "bnp": {
                "name": "BNP",
                "value": 420.0,
                "normal_range": {"min": 0, "max": 100},
                "captured_ago": "3 days ago",
                "capture_time": "2149-12-17T14:20:00",
            },
        },
        "recommendations": [
            "Cardiac risk is moderate with convergent calculators. Consider cardiology consultation.",
            "Elevated BNP (420 pg/mL) suggests heart failure risk. Optimize cardiac status pre-operatively.",
            "Anemia present (Hgb 11.7 g/dL). Consider pre-operative optimization if time permits.",
        ],
        "surgery": {
            "name": "Insertion of endotracheal tube",
            "type": "Thoracic",
            "expected_duration_minutes": 30,
            "duration_inferred": True,
        },
    }

    # Generate summary using free AI agent
    print("=" * 80)
    print("AI SUMMARIZER - Using Free AI Agent")
    print("=" * 80)
    print("\n")

    # Try to use local model first (completely free, no internet needed)
    print(
        "Attempting to generate summary using local AI model (free, no internet required)..."
    )
    print("(First run will download the model, subsequent runs are instant)\n")

    result = summarize_risk_assessment(
        sample_input, api_provider="local", use_mock=False
    )

    print("PROMPT SENT TO AI MODEL:")
    print("-" * 80)
    print(result["prompt"])
    print("\n")

    print("GENERATED SUMMARY (from AI agent):")
    print("-" * 80)
    print(result["summary"])
    print("\n")

    print("METADATA:")
    print("-" * 80)
    print(json.dumps(result["metadata"], indent=2))
    print("\n")

    print("=" * 80)
    print("USAGE INSTRUCTIONS:")
    print("=" * 80)
    print("\n1. Local Model (Free, recommended, no internet needed):")
    print(
        "   result = summarize_risk_assessment(risk_assessment, api_provider='local')"
    )
    print("   Note: First run downloads the model (~1.6GB), then works offline")
    print("\n2. Hugging Face API (Free, requires internet):")
    print(
        "   result = summarize_risk_assessment(risk_assessment, api_provider='huggingface')"
    )
    print("   Optional: Set HUGGINGFACE_API_KEY env var for faster/private models")
    print("\n3. Google Gemini (Free tier, requires API key and internet):")
    print("   export GOOGLE_AI_API_KEY='your-key-here'")
    print(
        "   result = summarize_risk_assessment(risk_assessment, api_provider='gemini')"
    )
    print("\n4. Mock mode (for testing without AI):")
    print("   result = summarize_risk_assessment(risk_assessment, use_mock=True)")
    print("=" * 80)
