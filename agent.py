import os
import argparse
import importlib.util
import sys
import traceback
import time
import textwrap
import pandas as pd
from pathlib import Path

try:
    from google import genai
except Exception:
    genai = None

MAX_ATTEMPTS = 3

# Helpers
def find_sample_files(target):
    base = Path("data") / target
    if not base.exists():
        raise FileNotFoundError(f"data directory not found: {base}")
    pdfs = list(base.glob("*.pdf"))
    csvs = list(base.glob("*.csv"))
    if len(pdfs) == 0 or len(csvs) == 0:
        raise FileNotFoundError(f"Need at least one .pdf and one .csv in {base}")
    return str(pdfs[0]), str(csvs[0])

def read_csv_preview(csv_path, n=5):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return df, df.head(n).to_dict(orient="records")

def write_module(path, code):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

def import_parse_function(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Unable to load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"No loader for spec {module}")
    loader.exec_module(module)
    if not hasattr(module, "parse"):
        raise AttributeError("Generated module does not define parse(pdf_path)")
    return module.parse

def normalize_df_for_compare(df, expected_df):
    
    df2 = df.copy()
    df2 = df2.reset_index(drop=True)
    
    for c in expected_df.columns:
        if c not in df2.columns:
            df2[c] = ""
    df2 = df2[expected_df.columns]
    
    df2 = df2.fillna("").astype(str).apply(lambda s: s.str.strip())
    exp = expected_df.fillna("").astype(str).apply(lambda s: s.str.strip())
    exp = exp.reset_index(drop=True)[df2.columns]
    return df2, exp

def run_parse_and_test(parser_path, pdf_path, csv_path, module_name="gen_parser"):
    # Import parse function
    parse_fn = import_parse_function(parser_path, module_name)
    expected_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # run parse
    result_df = parse_fn(pdf_path)
    if not isinstance(result_df, pd.DataFrame):
        raise TypeError("parse() did not return a pandas.DataFrame")
    # Normalize both and compare
    got, exp = normalize_df_for_compare(result_df, expected_df)
    equal = got.equals(exp)
    return equal, got, exp

def llm_is_available():
    return (genai is not None) and (os.environ.get("Gemini_API_KEY") is not None)

def clean_code_from_response(text):
    if text.startswith("```"):
       
        idx = text.find("\n")
        if idx != -1 and text[:3] == "```":
           
            if text[:6].startswith("```py") or text[:8].startswith("```python"):
                
                first_newline = text.find("\n")
                text = text[first_newline+1:]
        
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()

def call_genai_generate_module(system_prompt, user_prompt, model=None, max_tokens=1500):
    if genai is None:
        raise RuntimeError("genai package not installed")
    key = os.environ.get("Gemini_API_KEY")
    if not key:
        raise RuntimeError("Gemini_API_KEY not set")
    genai.api_key = key
    model = model or os.environ.get("Gemini_AI_MODEL", "2.5")
    # messages pattern
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": user_prompt}
    ]
    resp = genai.ChatCompletion.create(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens)
    text = resp["choices"][0]["message"]["content"]
    return clean_code_from_response(text)

# Heuristic parser generator

def make_heuristic_parser_code(target, expected_df):
    cols = list(expected_df.columns)
    cols_repr = repr(cols)
    # Template: tries pdfplumber table extraction, then simple cleanup and column mapping
    template = f'''\
import pandas as pd
import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Heuristic parser:
      - uses pdfplumber to extract page tables
      - concatenates found tables and attempts to map columns to the expected schema
    Expected columns: {cols_repr}
    This is a best-effort parser and may need manual tweaks.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                table = page.extract_table()
                if table and len(table) > 1:
                    # first row as header if it looks like header
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
            except Exception:
                continue
    if len(tables) == 0:
        raise ValueError("No tables found by pdfplumber. Heuristic parser failed.")
    df = pd.concat(tables, ignore_index=True, sort=False)
    # Simple cleanup:
    df = df.rename(columns=lambda s: s.strip() if isinstance(s, str) else s)
    # If column counts match expected, try direct rename:
    if df.shape[1] == {len(cols)}:
        df.columns = {cols_repr}
    else:
        # Attempt to map columns by fuzzy matching: try exact name, or substring match
        mapping = {{}}
        for c in df.columns:
            cstr = str(c).lower()
            for expected in {cols_repr}:
                if expected.lower() == cstr:
                    mapping[c] = expected
                    break
            else:
                for expected in {cols_repr}:
                    if expected.lower() in cstr or cstr in expected.lower():
                        mapping[c] = expected
                        break
        df = df.rename(columns=mapping)
        # ensure all expected columns exist
        for expected in {cols_repr}:
            if expected not in df.columns:
                df[expected] = ""
        df = df[{cols_repr}]
    # Final strip
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df
'''
    return textwrap.dedent(template)

# Main agent loop
def run_agent(target):
    print(f"[agent] target={target}")
    pdf_path, csv_path = find_sample_files(target)
    expected_df, preview = read_csv_preview(csv_path, n=5)

    module_path = Path("custom_parsers") / f"{target}_parser.py"
    module_path = str(module_path)

    # If LLM available -> use LLM loop
    if llm_is_available():
        print("[agent] GENAI API detected: using LLM generation loop")
        system_prompt = (
            "You are an assistant that writes a Python module file implementing a parser for a bank statement PDF.\n"
            "Produce only valid Python code for a module file (no explanations, no markdown). The module MUST define:\n"
            "def parse(pdf_path: str) -> pandas.DataFrame:  # returns a pandas DataFrame matching the provided CSV schema.\n"
            "Constraints:\n"
            " - The output DataFrame must have EXACT column names and order as given in the CSV sample.\n"
            " - Do not print or write files. Self-contained module is preferred.\n"
            " - Show necessary imports at top (pandas, pdfplumber, re, etc.) if used.\n"
            " - Keep code deterministic and robust; handle whitespace and common formatting issues.\n"
            "If you must use third-party packages (tabula, camelot) mention them but preference is for pdfplumber/pandas.\n"
        )
        base_user_prompt = f"""
Generate a Python module and implement parse(pdf_path) that reads a bank statement PDF and returns a pandas.DataFrame
that exactly matches the CSV sample. Here is the CSV preview (first rows) and columns:

CSV columns: {list(expected_df.columns)}
CSV head (first rows): {preview}

Constraints:
- Module file path: custom_parsers/{target}_parser.py
- Function signature: def parse(pdf_path) -> pandas.DataFrame
- Use pdfplumber or pandas; do not print or write files.
- The DataFrame returned must match column names and order exactly.

Provide only the Python module source.
"""
        prompt = base_user_prompt
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"[agent] Attempt {attempt}")
            if last_error:
                # ask LLM to produce a fixed version, show failing traceback + previous code
                user_prompt = prompt + "\nThe previous attempt failed with this Python traceback:\n" + last_error + "\nPlease produce a corrected module. Only output the module code."
            else:
                user_prompt = prompt
            try:
                code = call_genai_generate_module(system_prompt, user_prompt)
            except Exception as e:
                raise RuntimeError("LLM call failed: " + str(e))
            write_module(module_path, code)
            try:
                ok, got, exp = run_parse_and_test(module_path, pdf_path, csv_path, module_name=f"parser_{target}")
                if ok:
                    print(f"[agent] SUCCESS on attempt {attempt} — parser written to {module_path}")
                    return True
                else:
                    # prepare traceback-like info for next attempt
                    # show a small sample difference and repr of first few rows
                    diff_info = f"First rows expected:\n{exp.head(3).to_string(index=False)}\nFirst rows got:\n{got.head(3).to_string(index=False)}\n"
                    last_error = diff_info
                    print(f"[agent] Attempt {attempt} produced wrong DataFrame. Will retry.")
            except Exception as e:
                tb = traceback.format_exc()
                last_error = tb
                print(f"[agent] Attempt {attempt} raised exception during test:\n{tb}\nWill retry if attempts remain.")
        print("[agent] All attempts exhausted and parser did not pass tests.")
        return False
    else:
        # Heuristic fallback
        print("[agent] No GENAI_API_KEY found or genai package unavailable: using heuristic parser generator.")
        code = make_heuristic_parser_code(target, expected_df)
        write_module(module_path, code)
        try:
            ok, got, exp = run_parse_and_test(module_path, pdf_path, csv_path, module_name=f"parser_{target}")
            if ok:
                print(f"[agent] Heuristic parser succeeded: {module_path}")
                return True
            else:
                print(f"[agent] Heuristic parser did NOT match CSV. First rows expected:\n{exp.head(3)}\nFirst rows got:\n{got.head(3)}")
                return False
        except Exception as e:
            print("[agent] Heuristic parser raised an exception while testing:", str(e))
            return False

# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="target bank folder name under data/ (e.g., icici)")
    args = parser.parse_args()
    ok = run_agent(args.target)
    if not ok:
        sys.exit(2)

if __name__ == "__main__":
    main()
