"""Literal categorical matching; no regular expressions or executable filter expressions."""
# SETUP LOGIC: Use nullable text to preserve missing categories distinctly from literal strings.
import pandas as pd


def category_mask(source, values, *, match='exact', case_sensitive=True):
    # VALIDATION LOGIC: Partial matching is always literal, including punctuation and regex metacharacters.
    if match not in {'exact','contains','starts_with','ends_with'}:
        raise ValueError('match must be exact, contains, starts_with or ends_with.')
    if isinstance(values, (str, bytes)):
        values = [values]
    # CORE LOGIC: STEP 1 — Normalize case only when explicitly requested.
    # Input: source=['Alpha Ltd','BETA',None], values=['alpha'], case_sensitive=False.
    # Output: text=['alpha ltd','beta',<NA>], terms=['alpha'].
    # Explanation: Missing values stay missing and cannot match the literal string 'None'.
    # Trick: casefold handles Unicode case consistently; no trimming alters the stored category values.
    text = source.astype('string')
    terms = [str(v) for v in values]
    if not case_sensitive:
        text, terms = text.str.casefold(), [v.casefold() for v in terms]
    # CONFIGURATION LOGIC: Select a literal pandas operation without executing user-supplied expressions.
    operations = {'exact':text.eq, 'contains':lambda term:text.str.contains(term, regex=False),
                  'starts_with':text.str.startswith, 'ends_with':text.str.endswith}
    # CORE LOGIC: STEP 2 — Combine requested values with OR while excluding missing records.
    # Input: text=['alpha ltd','beta',<NA>], terms=['alpha','bet'], match='contains'.
    # Output: mask=[True,True,False].
    # Explanation: Either substring is enough; an empty list matches no records.
    # Trick: regex=False means '.' matches a literal dot, not an arbitrary character.
    mask = pd.Series(False, index=source.index)
    for term in terms:
        mask |= operations[match](term).fillna(False)
    return mask
