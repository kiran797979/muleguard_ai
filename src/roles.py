"""
MuleGuard AI — role based column resolution.

The problem this solves:

    Stage 2 asks for `TOT_TXNAMT_CR_L7D`. On the hackathon file that name exists.
    On somebody else's extract the same quantity might be called
    `credit_value_week`, `InwardAmt7Day`, or `sum.amt.in.7d`. Exact matching
    finds none of them, so the 29 behavioural features silently fail to build
    and the pipeline falls back to generic aggregates.

    That matters more than it sounds. The feature ablation in
    `08_feature_ablation.py` shows the behavioural features are the part of the
    signal LEAST explained by this dataset's extract artefact, so they are the
    part most likely to transfer to real data. Losing them on an unfamiliar
    schema loses the thing worth keeping.

The approach: stop matching names and start matching MEANING. Every column name
in retail banking encodes the same handful of ideas:

    STAT       total, average, maximum, minimum, ratio, deviation
    MEASURE    an amount, a count of transactions, or a balance
    DIRECTION  money in, money out, or unspecified
    WINDOW     7 days, 14 days, a month
    CHANNEL    cash, UPI, ATM, cheque, card, online transfer …

Decompose both the request and the available columns into that tuple, then match
tuple to tuple. `TOT_TXNAMT_CR_L7D` and `sum.amt.in.7d` decompose identically, so
one resolves the other without either name being known in advance.

Nothing here guesses. When no column carries the requested role the lookup
returns None and Stage 2 records the miss, exactly as it does for an exact-name
failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# --------------------------------------------------------------------------
# Vocabularies. Deliberately broad: these are the words retail banking extracts
# actually use, across vendors and in-house schemas.
# --------------------------------------------------------------------------
MEASURE_TOKENS = {
    "amount": {"AMT", "AMOUNT", "AMNT", "AMTS", "VAL", "VALUE", "VOL", "VOLUME",
               "SUMAMT", "TXNAMT", "TRANAMT", "MONEY", "RUPEES", "INR"},
    "count": {"TXN", "TXNS", "TRAN", "TRANS", "TRANSACTION", "TRANSACTIONS",
              "CNT", "COUNT", "NUM", "NUMBER", "FREQ", "FREQUENCY", "HITS"},
    "balance": {"BAL", "BALANCE", "BALANCES", "EOD", "CLOSINGBAL", "AVGBAL"},
}

DIRECTION_TOKENS = {
    "credit": {"CR", "CRED", "CREDIT", "CREDITS", "IN", "INW", "INWARD", "INFLOW",
               "RECD", "RECEIVED", "RECEIPT", "DEP", "DEPOSIT", "DEPOSITS", "INCOMING"},
    "debit": {"DB", "DR", "DEB", "DEBIT", "DEBITS", "OUT", "OUTW", "OUTWARD",
              "OUTFLOW", "SENT", "WDL", "WD", "WITHDRAWAL", "WITHDRAWALS",
              "WITHDRAWN", "PAID", "PAYMENT", "OUTGOING"},
}

# Windows are stored as a day count so "month", "31D" and "30D" unify.
WINDOW_TOKENS = {
    7: {"7D", "L7D", "L7", "7DAY", "7DAYS", "D7", "WK", "WEEK", "WEEKLY", "LAST7"},
    14: {"14D", "L14D", "L14", "14DAY", "14DAYS", "D14", "FORTNIGHT", "2WK"},
    31: {"31D", "30D", "L31D", "L30D", "L31", "L30", "31DAY", "30DAY", "30DAYS",
         "31DAYS", "D30", "D31", "MTD", "MONTHLY", "1M", "M1"},
    90: {"90D", "L90D", "90DAY", "QTR", "QUARTER", "3M"},
}

CHANNEL_TOKENS = {
    "CASH": {"CASH", "CSH"},
    "UPI": {"UPI"},
    "ATM": {"ATM"},
    "CHQ": {"CHQ", "CHEQUE", "CHECK", "CHK"},
    "ELEC_XFER": {"ELEC", "XFER", "TRANSFER", "TRF", "NEFT", "IMPS", "RTGS",
                  "WIRE", "ELECXFER", "FUNDTRANSFER"},
    "POS_PYMT": {"POS", "CARD", "MERCHANT", "SWIPE", "PYMT"},
    "NET_BNKING": {"NETBNKING", "NETBANKING", "NETBNK", "INTERNET", "IBANKING", "WEB"},
    "MOBILE": {"MBNKING", "MOBILE", "MBANK", "APP"},
    "APB": {"APB", "AADHAAR", "AEPS"},
    "BBPS": {"BBPS", "BILL", "BILLPAY", "UTILITY"},
    "GST": {"GST", "TAX"},
}

STAT_TOKENS = {
    "TOT": {"TOT", "TOTAL", "SUM", "AGG", "AGGREGATE"},
    "AVG": {"AVG", "AVERAGE", "MEAN"},
    "MAX": {"MAX", "MAXIMUM", "PEAK", "HIGH"},
    "MIN": {"MIN", "MINIMUM", "LOW"},
    "RATIO": {"R", "RA", "RT", "RATIO", "PCT", "PERCENT", "SHARE"},
    "DEV": {"D", "DA", "DEV", "DEVIATION", "ZSCORE", "Z"},
}

# Words that must never be read as a window. `MNTH` is the extract-month column
# on the hackathon file: reading it as "a 31 day window" would quietly reinstate
# the exact confound the pipeline exists to remove.
WINDOW_BLOCKLIST = {"MNTH", "MONTH", "MON", "MTH"}

# Identifier guard. `account_number` contains NUMBER, which sits in the count
# vocabulary, so without this an account id parses as "a count of things" and
# could be resolved into a behavioural feature. Feeding a row key into a
# pass-through ratio is exactly the kind of silent wrongness that survives
# review, so identifiers are rejected before any other inference runs.
ID_SUBJECT = {"ACCOUNT", "ACCT", "AC", "CUST", "CUSTOMER", "CLIENT", "PARTY",
              "REF", "ROW", "RECORD", "USER", "MEMBER"}
ID_QUALIFIER = {"NUMBER", "NUM", "NO", "ID", "KEY", "CODE", "IDENT", "IDENTIFIER",
                "SRNO", "SLNO", "UID"}


@dataclass(frozen=True)
class Role:
    """What a column means, independent of what it is called."""
    measure: str | None = None      # amount / count / balance
    direction: str | None = None    # credit / debit / None
    window: int | None = None       # 7 / 14 / 31 / 90
    channel: str | None = None      # CASH / UPI / …
    stat: str | None = None         # TOT / AVG / MAX / MIN / RATIO / DEV

    def specificity(self) -> int:
        return sum(x is not None for x in
                   (self.measure, self.direction, self.window, self.channel, self.stat))


# Split camelCase on lowercase-to-uppercase ONLY. Including digits here (the
# obvious first attempt) breaks every window token: `AVG_BAL_7DAYS` becomes
# `7` + `DAYS`, `7D` becomes `7` + `D`, and the window is then invisible.
_CAMEL = re.compile(r"([a-z])([A-Z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_NUMSPLIT = re.compile(r"(\d+)")


def tokenise(name: object) -> list[str]:
    """Break a column name into comparable word pieces.

    Handles snake_case, kebab-case, dotted, spaced and camelCase, and keeps
    digit runs attached to their suffix so `7D` survives as one token while
    `L7D` also yields `7D`.
    """
    s = _CAMEL.sub(r"\1 \2", str(name))
    parts = [p.upper() for p in _SPLIT.split(s) if p]
    out: list[str] = []
    for p in parts:
        out.append(p)
        # `L7D` -> also offer `7D`; `AMOUNT7DAY` -> also `7DAY`
        chunks = [c for c in _NUMSPLIT.split(p) if c]
        if len(chunks) > 1:
            out.extend(chunks)
            for i in range(len(chunks) - 1):
                if chunks[i].isdigit():
                    out.append(chunks[i] + chunks[i + 1])

    # A separator can also sit between the number and its unit, as in
    # "total inward amount 31 days" or "credit_amt_7_d". Rejoin those, or the
    # window is split across two tokens and never recognised.
    for a, b in zip(parts, parts[1:]):
        if a.isdigit():
            out.append(a + b)
    return out


def _lookup(tokens: list[str], table: dict) -> object | None:
    """First table key whose vocabulary contains any of these tokens."""
    tset = set(tokens)
    for key, vocab in table.items():
        if tset & vocab:
            return key
    return None


@lru_cache(maxsize=8192)
def parse_role(name: str) -> Role:
    """Decompose a column name into what it actually measures."""
    toks = tokenise(name)
    tset = set(toks)

    # An identifier is not a measurement, whatever words it happens to contain.
    if (tset & ID_SUBJECT) and (tset & ID_QUALIFIER):
        return Role()

    measure = _lookup(toks, MEASURE_TOKENS)
    # A name can carry both an amount and a count word (TXNAMT). Amount wins,
    # because that is what the compound actually denotes.
    if measure == "count" and tset & MEASURE_TOKENS["amount"]:
        measure = "amount"

    direction = _lookup(toks, DIRECTION_TOKENS)

    window = None
    if not (tset & WINDOW_BLOCKLIST):
        window = _lookup(toks, WINDOW_TOKENS)

    channel = _lookup(toks, CHANNEL_TOKENS)

    # Single-letter stat prefixes only count in leading position, otherwise a
    # stray "D" anywhere in a name would mark every column as a deviation.
    stat = None
    for key, vocab in STAT_TOKENS.items():
        hit = tset & vocab
        if not hit:
            continue
        if any(len(h) <= 2 for h in hit) and toks and toks[0] not in vocab:
            continue
        stat = key
        break

    return Role(measure=measure, direction=direction, window=window,
                channel=channel, stat=stat)


class RoleIndex:
    """Every column in a dataset, indexed by what it means."""

    def __init__(self, columns, label_of=None):
        """`label_of` maps a column to its readable name when a dictionary exists."""
        self.roles: dict[str, Role] = {}
        for c in columns:
            name = label_of(c) if label_of else c
            self.roles[c] = parse_role(str(name))

    def find(self, want: Role, exclude: set | None = None) -> str | None:
        """Best column carrying this role, or None.

        Scoring is deliberately strict on the parts that change the meaning of a
        feature and lenient on the parts that only change its flavour. Measure
        and direction must match exactly when requested, because a credit total
        and a debit total are not interchangeable and silently swapping them
        would invert a pass-through ratio.
        """
        exclude = exclude or set()
        best, best_score = None, 0

        for col, role in self.roles.items():
            if col in exclude:
                continue
            if want.measure and role.measure != want.measure:
                continue
            if want.direction and role.direction != want.direction:
                continue
            # An undirected request must not silently pick up a one-sided column.
            if want.direction is None and role.direction is not None:
                continue
            if want.channel and role.channel != want.channel:
                continue
            if want.channel is None and role.channel is not None:
                continue
            if want.window and role.window != want.window:
                continue

            score = 4
            if want.stat and role.stat == want.stat:
                score += 2
            elif want.stat and role.stat is not None:
                score -= 1
            if want.window and role.window == want.window:
                score += 1

            if score > best_score:
                best, best_score = col, score

        return best


def describe(name: str) -> str:
    """Human readable role, for reports."""
    r = parse_role(name)
    bits = [b for b in (
        r.stat, r.measure, r.direction,
        f"{r.window}d" if r.window else None, r.channel) if b]
    return " · ".join(bits) if bits else "unclassified"
