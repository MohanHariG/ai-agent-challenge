# ai-agent-challenge
Coding agent challenge which write custom parsers for Bank statement PDF.

## 5-step run instructions
1. Clone repository and cd into it.
2. Create a virtualenv and install dependencies:
   - python -m venv venv && source venv/bin/activate
   - pip install -r requirements.txt

3. Put a sample PDF and CSV into data/<target>/ (e.g., data/icici/icici_sample.pdf and data/icici/icici_sample.csv).

4. (Optional) Set OpenAI API key:
    - export OPENAI_API_KEY="sk-..."
5. Run the agent:
   - python agent.py --target icici
