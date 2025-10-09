import pandas as pd
import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Heuristic parser:
      - uses pdfplumber to extract page tables
      - concatenates found tables and attempts to map columns to the expected schema
    Expected columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
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
    if df.shape[1] == 5:
        df.columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    else:
        # Attempt to map columns by fuzzy matching: try exact name, or substring match
        mapping = {}
        for c in df.columns:
            cstr = str(c).lower()
            for expected in ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']:
                if expected.lower() == cstr:
                    mapping[c] = expected
                    break
            else:
                for expected in ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']:
                    if expected.lower() in cstr or cstr in expected.lower():
                        mapping[c] = expected
                        break
        df = df.rename(columns=mapping)
        # ensure all expected columns exist
        for expected in ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']:
            if expected not in df.columns:
                df[expected] = ""
        df = df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]
    # Final strip
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df
