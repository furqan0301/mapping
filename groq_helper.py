import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def explain_solution_groq(problem_text: str, solution_text: str, api_key: str, model: str = "llama3-8b-8192") -> str:
    if not api_key:
        return "Groq API key not set. Add GROQ_API_KEY in Streamlit secrets."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful tutor. Explain CSP map-coloring solution clearly for a beginner."},
            {"role": "user", "content": f"Problem:\n{problem_text}\n\nSolution:\n{solution_text}\n\nExplain steps and why constraints are satisfied."}
        ],
        "temperature": 0.3
    }

    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"]["message"]["content"]
