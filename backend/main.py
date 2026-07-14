from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sqlite3
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.responses import StreamingResponse
from groq import Groq, RateLimitError
from dotenv import load_dotenv
import os
import re
import json
import math

# LangSmith tracing
from langsmith import traceable
from langsmith.run_helpers import trace

load_dotenv()



# step 1 — load data into SQLite
print("creating a sqlite database and loading the data")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "docs", "titanic_cleaned_new.csv")
df = pd.read_csv(csv_path)
conn = sqlite3.connect("titanics.db", check_same_thread=False)
df.to_sql("titanic", conn, if_exists="replace", index=False)
conn.commit()
print("data is loaded into db")

# step 2 — build schema descriptions
col_description = {
    "Age": "Represents how old each passenger was, in years. Relevant for questions about oldest, youngest, elderly, senior, minors, children, or age comparisons.",
    "Fare": "The ticket price paid by each passenger. Relevant for questions about rich, poor, expensive, cheap, wealthy, or cost comparisons.",
    "Sex": "The gender of each passenger, male or female. Relevant for questions about men, women, gender splits.",
    "Pclass": "The passenger's ticket class (1st, 2nd, 3rd). Relevant for questions about class, cabin tier, first class, economy.",
    "SibSp": "Number of siblings or spouses aboard with the passenger. Relevant for questions about family size, siblings, spouses.",
    "Parch": "Number of parents or children aboard with the passenger. Relevant for questions about family, parents, children traveling together.",
    "Survived": "Whether the passenger survived (1) or died (0). Relevant for questions about survivors, deaths, casualties, fatalities.",
    "Embarked": "The port where the passenger boarded (S=Southampton, C=Cherbourg, Q=Queenstown). Relevant for questions about boarding location, port.",
}
def build_schema(df, col):
    col_values = df[col]
    dtype = str(col_values.dtype)

    if col_values.nunique() <= 12:
        values = sorted(col_values.dropna().unique().tolist())
        desc= f"column: {col}, dtype: {dtype}, unique values: {values}"
    elif dtype in ["int64", "float64"]:
        desc= f"column: {col}, dtype: {dtype}, min: {col_values.min()}, max: {col_values.max()}, mean: {col_values.mean():.2f}"
    else:
        sample = col_values.dropna().sample(min(5, len(col_values)), random_state=42).tolist()
        desc= f"column: {col}, dtype: {dtype}, sample values: {sample}"
    description=col_description.get(col,"")
    if description:
        desc+=f".{description}"
    print("schema building is done")
    return desc

schema = {col: build_schema(df, col) for col in df.columns}
column_dtypes = {col: str(df[col].dtype) for col in df.columns}
print(schema)

# step 3 — query classification (sql vs general)
groq_model = os.getenv("GROQ_MODEL")
groq_small_model=os.getenv("groq_model2")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)



DESTRUCTIVE_PATTERN = re.compile(
    r"\b(delete|remove|drop|update|modify|change|insert|add|alter|truncate|mark)\b"
    r"|\bset\s+\w+\s*(=|to)\b"
    r"|\bto\s+(not\s+survived|survived|dead|alive|deceased)\b",
    re.IGNORECASE
)

# NOT traced — pure regex, no LLM/retrieval involved
def input_level_guardrail(question: str) -> bool:
    return bool(DESTRUCTIVE_PATTERN.search(question))


@traceable(name="standalone_question", run_type="llm")
def standalone_question(query: str, history: list[dict]) -> str:
    if not history:
        return query
    context = []
    for turn in history[-5:]:
        line = f"Question:{turn['question']}"
        if turn.get('sql'):
            line += f"\nSQL: {turn['sql']}"
        if turn.get('result_summary'):
            line += f"\n Result:{turn['result_summary']}"
        context.append(line)
    context = "\n---\n".join(context)
    prompt = f"""conversation history:{context}
    new question:{query}
    Rewrite the new message as a fully standalone question, resolving pronouns
    or references to previous turns (e.g. "that", "their", "what about X","where are they").
    Also correct any grammar, spelling, or phrasing issues so the question reads naturally
    and clearly, without changing its meaning or intent.
    If already standalone and grammatically correct, return unchanged.
    return only the rewritten question nothing else"""
    response = groq_client.chat.completions.create(
        model=groq_small_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_effort="low"
    )
    print("standalone question:",response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


@traceable(name="classify_query", run_type="llm")
def classify_query(query):
    response = groq_client.chat.completions.create(
        model=groq_small_model,
        messages=[{
            "role": "user",
            "content": f"""Classify this question as either 'sql' or 'general'.

'sql' = questions that need data from the Titanic dataset
Examples: counts, averages, lists, filters, rates, comparisons, top N, survival stats
"give me full/complete details of X", "show me everything about X", "list all information about X"
— any request for actual passenger records or row-level data counts as 'sql', even without
an aggregate word like count/average.


'general' = factual or explanation questions that don't need data
Examples: what is Pclass, explain SibSp, what was the Titanic, what does Fare mean

Reply with ONLY one word: sql or general
No explanation. No punctuation. Just the word.

Question: {query}"""
        }],
        temperature=0,
        reasoning_effort="low"

    )
    result = response.choices[0].message.content.strip().lower()
    return "sql" if "sql" in result else "general"

# step 4 — embed schemas into chromadb
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("titanic_collections")
except Exception:
    pass

collections = client.create_collection("titanic_collections", metadata={"hnsw:space": "cosine"})

cols = list(schema.keys())
descriptions = list(schema.values())
embeddings = embed_model.encode(descriptions).tolist()

collections.add(ids=cols, documents=descriptions, embeddings=embeddings)
print("embedded and stored in chromadb")

# step 4b — keyword + semantic retrieval
def tokenize(text):
    # strip punctuation so "['female'," tokenizes as "female"
    return re.findall(r"[a-z0-9]+", text.lower())

all_chunks = collections.get()
tokenized_docs = [tokenize(doc) for doc in all_chunks['documents']]
bm25 = BM25Okapi(tokenized_docs)
col_ids = all_chunks["ids"]

# NOT traced individually — these are cheap, sub-steps of final_schema_ranking,
# which IS traced below. Tracing every sub-call here would clutter the tree.
def keyword_search(query, top_k=5):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [col_ids[i] for i in top_indices]

def semantic(query, top_k=5):
    query_embedding = embed_model.encode(query).tolist()
    results = collections.query(query_embeddings=[query_embedding], n_results=top_k)
    return list(results["ids"][0])

def rrf_fuse(list1, list2, k=60):
    scores = {}
    for rank, item in enumerate(list1):
        scores[item] = scores.get(item, 0) + 1 / (k + rank)
    for rank, item in enumerate(list2):
        scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

COLUMN_SYNONYMS = {
    "Age": ["age", "old", "young", "younger", "older", "oldest", "youngest", "elderly", "senior", "minor", "child", "children", "kid", "kids", "infant", "adult"],
    "Fare": ["fare", "price", "cost", "paid", "ticket price", "expensive", "cheap", "rich", "poor", "wealthy", "wealth"],
    "Sex": ["sex", "gender", "male", "female", "men", "women", "man", "woman", "boys", "girls"],
    "Pclass": ["class", "pclass", "ticket class", "first class", "second class", "third class", "cabin tier", "economy"],
    "SibSp": ["sibsp", "sibling", "siblings", "spouse", "spouses", "brother", "sister", "husband", "wife"],
    "Parch": ["parch", "parent", "parents", "children", "child", "kids", "family", "families"],
    "Survived": ["survived", "survival", "survivor", "survivors", "died", "death", "deaths", "dead", "alive", "perished", "casualty", "casualties", "fatality", "fatalities"],
    "Embarked": ["embarked", "embark", "boarded", "boarding", "port", "southampton", "cherbourg", "queenstown"],
    "Name": ["name", "names", "named", "who is", "who was"],
    "Cabin": ["cabin", "cabins", "room", "deck", "berth"],
    "PassengerId": ["passenger id", "passengerid", "id number"],
    "Ticket": ["ticket number", "ticket id"],
}


def hint_columns_from_query(query: str, all_columns: list[str]) -> list[str]:
    q = query.lower()
    hits = []

    # 1. exact column name match (existing logic)
    for col in all_columns:
        pattern = r"\b" + re.escape(col.lower()) + r"\b"
        if re.search(pattern, q):
            hits.append(col)

    # 2. synonym match — catches phrasing that doesn't literally name the column
    for col, synonyms in COLUMN_SYNONYMS.items():
        if col in all_columns and col not in hits:
            for synonym in synonyms:
                if " " in synonym:
                    # multi-word phrase — plain substring check
                    if synonym in q:
                        hits.append(col)
                        break
                else:
                    # single word — word-boundary match to avoid partial matches
                    if re.search(r"\b" + re.escape(synonym) + r"\b", q):
                        hits.append(col)
                        break

    return hits


@traceable(name="final_schema_ranking", run_type="retriever")
def final_schema_ranking(query, top_k=5):
    semantic_ranks = semantic(query)
    keyword_ranks = keyword_search(query)
    fused = rrf_fuse(semantic_ranks, keyword_ranks)
    ranked = [col for col, score in fused[:top_k]]
    forced = hint_columns_from_query(query, list(schema.keys()))
    for col in forced:
        if col not in ranked:
            ranked.append(col)
    return ranked

from few_shot_examples import FEW_SHOT_EXAMPLES
client = chromadb.PersistentClient(path="./chroma_few_shot_db")
try:
    client.delete_collection("few_shot_examples")
except Exception:
    pass
few_shot_collection = client.create_collection("few_shot_examples", metadata={"hnsw:space": "cosine"})
ids = [str(i) for i in range(len(FEW_SHOT_EXAMPLES))]
questions = [ex["question"] for ex in FEW_SHOT_EXAMPLES]
embeddings = embed_model.encode(questions).tolist()
few_shot_collection.add(ids=ids, documents=questions, embeddings=embeddings)


@traceable(name="get_few_shot_examples", run_type="retriever")
def get_few_shot_examples(query, top_k=2):
    query_embeddings = embed_model.encode(query).tolist()
    results = few_shot_collection.query(query_embeddings=[query_embeddings], n_results=top_k)
    blocks = []
    for ids_str in results["ids"][0]:
        ex = FEW_SHOT_EXAMPLES[int(ids_str)]
        blocks.append(f"Question: {ex['question']}\nSQL:\n{ex['sql']}")
        print(f"the top k few shot examples are :{blocks}")
    return "\n---\n".join(blocks)

# intent classification prompt — NOT traced, pure string building
def build_intent_prompt(retrieved_schema: list[str], examples, error_feedback: str = "") -> str:
    schema_text = "\n".join(f"- {schema[col]}" for col in retrieved_schema)
    return f"""You are an intent classifier for a SQL chatbot querying the titanic table.
Extract the user's question into this exact JSON structure. Respond with ONLY the JSON —
no explanation, no markdown code fences, no text before or after.

{{
  "aggregate": "AVG" | "SUM" | "COUNT" | "MIN" | "MAX" | "RATE" | null,
  "column": "<column being aggregated, or '*' for row lookups>",
  "table": "titanic",
  "filter_groups": [
    {{
      "logic": "AND" | "OR",
      "conditions": [
        {{"column": "<col>", "operator": "=|>|<|>=|<=|!=|LIKE|IN", "value": "<value>"}}
      ]
    }}
  ],
  "rate_target": {{"column": "<col>", "operator": "=|>|<|>=|<=|!=", "value": "<value>"}} | null,
  "group_by": "<column>" | null,
  "having": {{"column": "<col>", "aggregate": "<agg>", "operator": "<op>", "value": "<val>"}} | null,
  "order_by": {{"column": "<col>", "direction": "ASC" | "DESC"}} | null,
  "limit": <int> | null,
  "distinct": true | false
}}
FILTER GROUPS RULE:
- Each item in "filter_groups" is one bracketed condition group.
- Conditions WITHIN a group combine using that group's own "logic".
- Groups ALWAYS combine with AND between each other, regardless of each group's internal logic.
- Put a single required condition in its own group with "logic": "AND".
- Put mutually exclusive alternatives (e.g. "class 1 or 2") together in one group with "logic": "OR".
Example:
Question: "male passengers in first or second class who survived"
Output: {{
  "aggregate": "COUNT", "column": "*", "table": "titanic",
  "filter_groups": [
    {{"logic": "AND", "conditions": [{{"column": "Sex", "operator": "=", "value": "male"}}]}},
    {{"logic": "OR", "conditions": [
        {{"column": "Pclass", "operator": "=", "value": "1"}},
        {{"column": "Pclass", "operator": "=", "value": "2"}}
    ]}},
    {{"logic": "AND", "conditions": [{{"column": "Survived", "operator": "=", "value": "1"}}]}}
  ],
  "rate_target": null, "group_by": null, "having": null,
  "order_by": null, "limit": null, "distinct": false
}}
STRICT RULES:
-"Who"/"which passenger" questions want the actual row (use order_by + limit), not an aggregate — even with words like "highest"/"oldest".
- Use ONLY columns listed below. Never use a column not listed, even if you know it exists in a typical Titanic dataset.
- If the question mentions gender (male, female, men, women), you MUST filter on Sex if it appears below.
- Every comparison/condition in the question (e.g. "above 20", "at least", "under") MUST produce a corresponding entry in "filters". Do not silently drop a condition.
- If a needed column isn't listed, set "column" to null and don't invent a substitute filter.

- If the user asks for specific named columns (e.g. "name and cabin", "show me age and fare"),
  set "column" to a comma-separated string of those exact column names, e.g. "Name, Cabin".
  Only use "*" when the user asks for "complete/full details" WITHOUT naming specific columns.

- RATE queries: if the question asks for a "rate", "percentage", "proportion", or "ratio" of some outcome
  (e.g. "rate of passengers survived", "% of male passengers who survived"), set "aggregate" to "RATE".
  Put the OUTCOME condition (e.g. Survived = 1) in "rate_target", NOT in "filters". Put any population-
  restricting conditions (e.g. "above age 20", "male passengers") in "filters" as usual — these define
  the denominator group. "rate_target" defines the numerator. Example:
  Question: "rate of male passengers survived above age 20, split by sex"
  Output: {{"aggregate": "RATE", "column": null, "table": "titanic",
  "filters": [{{"column": "Age", "operator": ">", "value": "20"}}],
  "rate_target": {{"column": "Survived", "operator": "=", "value": "1"}},
  "logic": "AND", "group_by": "Sex", "having": null, "order_by": null, "limit": null, "distinct": false}}
Available columns:
{schema_text}
Examples to understand the query and the appropriate answer:
{examples}
IMPORTANT: Respond with ONLY the JSON object described above. Do not include any
explanation, reasoning, markdown code fences, or text before or after the JSON."""

@traceable(name="call_llm_for_intent", run_type="llm")
def call_llm_for_intent(user_question: str, retrieved_schema: list[str], examples, error_feedback: str = "", previous_intent: dict = None) -> dict:
    if error_feedback and previous_intent:
        system_prompt = f"""your previous JSON output was invalid:
{json.dumps(previous_intent)}

Fix only these issues and return corrected JSON matching the same schema.
Errors found:
{error_feedback}

Valid columns: {list(schema.keys())}
Respond with ONLY the JSON object. No explanation, no markdown code fences, no text before or after."""
    else:
        system_prompt = build_intent_prompt(retrieved_schema, examples, error_feedback)

    response = groq_client.chat.completions.create(
        model=groq_small_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.2,
        max_tokens=1024,
        reasoning_format="hidden" 
    )

    raw_text = response.choices[0].message.content.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        intent = json.loads(raw_text)
    except json.JSONDecodeError:
        # fallback: model added extra text around the JSON — extract it manually
        intent = extract_json_from_text(raw_text)
        if intent is None:
            raise ValueError(f"LLM did not return valid JSON: {raw_text}")

    print(intent)
    return intent
def stream_groq_text(messages, model=None, temperature=0, reasoning_effort=None, reasoning_format=None):
    """Yields text chunks as they're generated by Groq, instead of waiting for the full response."""
    kwargs = {
        "model": model or groq_small_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_format:
        kwargs["reasoning_format"] = reasoning_format
    stream = groq_client.chat.completions.create(**kwargs)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
def extract_json_from_text(text: str) -> dict | None:
    """Fallback: find a JSON object embedded in extra text using brace matching."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None
#even tho we give the description to the llm , but if we mention female passengers the llm may forgot to add sex column
NUMERIC_HINT_PATTERN = re.compile(
    r"\b(above|below|over|under|greater than|less than|at least|at most|more than|fewer than)\b\s*(\d+)|(\d+)\s*(or more|or less|or older|or younger)",
    re.IGNORECASE
)
GENDER_PATTERN = re.compile(r"\b(male|female|men|women|man|woman)\b", re.IGNORECASE)
CLASS_PATTERN = re.compile(r"\b(first|second|third)\s*class\b", re.IGNORECASE)
SURVIVAL_PATTERN = re.compile(r"\b(survived|survivor|survivors|died|dead|perished|alive)\b", re.IGNORECASE)

# NOT traced — deterministic validation logic, no LLM call
def validate_intent(intent: dict, user_question: str, valid_columns: list[str]) -> list[str]:
    errors = []
    all_conditions = [
        f for group in (intent.get("filter_groups") or []) for f in group["conditions"]
    ]
    referenced_cols = set()#take out the column present in all the filters
    if intent.get("column") and intent["column"] != "*":
        for col in intent["column"].split(","):
            referenced_cols.add(col.strip())
    for f in all_conditions:
        referenced_cols.add(f["column"])
    if intent.get("group_by"):
        referenced_cols.add(intent["group_by"])
    if intent.get("order_by"):
        referenced_cols.add(intent["order_by"]["column"])
    if intent.get("having"):
        referenced_cols.add(intent["having"]["column"])
    if intent.get("rate_target"):
        referenced_cols.add(intent["rate_target"]["column"])

    for col in referenced_cols:#check if the column name is right
        if col not in valid_columns:
            errors.append(f"Column '{col}' does not exist in the schema. Valid columns: {valid_columns}")

    if intent.get("aggregate") == "RATE" and not intent.get("rate_target"):#if the aggregate is rate then the rate_target must be present
        errors.append("aggregate is 'RATE' but 'rate_target' is missing — add the outcome condition there.")

    for f in all_conditions:
        col, op = f["column"], f["operator"] #make sure the right operations are mapped to the right columns
        if col in column_dtypes and op in (">", "<", ">=", "<="):
            if column_dtypes[col] not in ("int64", "float64"):
                errors.append(
                    f"Filter uses numeric operator '{op}' on non-numeric column '{col}' (dtype={column_dtypes[col]})."
                )
 
    if NUMERIC_HINT_PATTERN.search(user_question):#any word in user ques matches  the words in numeric pattern
        has_numeric_filter = any(f["operator"] in (">", "<", ">=", "<=") for f in all_conditions)
        if not has_numeric_filter: #check any operator is present for this numeric question
            errors.append(
                "The question contains a numeric comparison (e.g. 'above 20') but no matching "
                "comparison filter (>, <, >=, <=) was found in 'filter_groups'. Add it."
            )
    if GENDER_PATTERN.search(user_question):
        has_sex_filter = any(f["column"] == "Sex" for f in all_conditions)
        if not has_sex_filter and intent.get("group_by") != "Sex":
            errors.append(
                "The question mentions gender (male/female/men/women) but no 'Sex' condition "
                "was found in 'filter_groups'. Add it."
            )
 
    if CLASS_PATTERN.search(user_question):
        has_class_filter = any(f["column"] == "Pclass" for f in all_conditions)
        if not has_class_filter and intent.get("group_by") != "Pclass":
            errors.append(
                "The question mentions a specific class (first/second/third) but no 'Pclass' "
                "condition was found in 'filter_groups'. Add it."
            )
 
    if SURVIVAL_PATTERN.search(user_question):
        has_survived_filter = any(f["column"] == "Survived" for f in all_conditions)
        has_survived_rate = intent.get("rate_target") and intent["rate_target"].get("column") == "Survived"
        if not has_survived_filter and not has_survived_rate:
            errors.append(
                "The question mentions survival/death but no 'Survived' condition was found "
                "in 'filter_groups' or 'rate_target'. Add it."
            )

    return errors
  


@traceable(name="classify_intent")
def classify_intent(user_question: str, retrieved_schema: list[str], examples, max_retries: int = 2) -> dict:
    error_feedback = ""
    last_intent = None

    for attempt in range(max_retries + 1):
        intent = call_llm_for_intent(
            user_question, retrieved_schema, examples,
            error_feedback=error_feedback,
            previous_intent=last_intent
        )
        last_intent = intent

        errors = validate_intent(intent, user_question, retrieved_schema)
        if not errors:
            return intent

        print(f"[validation attempt {attempt+1}] issues found: {errors}")
        error_feedback = "\n".join(f"- {e}" for e in errors)

    print(f"[warning] returning intent with unresolved validation issues: {error_feedback}")
    return last_intent
# generating sql using the intent — NOT traced, deterministic template filling
SQL_TEMPLATE = "SELECT {select} FROM {table}{joins}{where}{group_by}{having}{order_by}{limit}"

def fill_sql_template(intent: dict) -> str:
    slots = {
        "select": "*",
        "table": intent["table"],
        "joins": "",
        "where": "",
        "group_by": "",
        "having": "",
        "order_by": "",
        "limit": ""
    }

    if intent.get("aggregate") == "RATE":
        rt = intent.get("rate_target")
        if not rt:
            raise ValueError("aggregate is RATE but 'rate_target' is missing from intent")
        val = rt["value"]
        if isinstance(val, str) and not val.replace('.', '', 1).isdigit():
            val = f"'{val}'"
        condition = f"{rt['column']} {rt['operator']} {val}"
        slots["select"] = (
            f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS rate"
        )
    elif intent.get("aggregate"):
        agg_col = intent.get("column") or "*"
        slots["select"] = f"{intent['aggregate']}({agg_col})"
    elif intent.get("distinct") and intent.get("column"):
        slots["select"] = f"DISTINCT {intent['column']}"
    elif intent.get("column"):
        slots["select"] = intent["column"]

    if intent.get("group_by"):
        slots["select"] = f"{intent['group_by']}, {slots['select']}"
        slots["group_by"] = f" GROUP BY {intent['group_by']}"

    if intent.get("joins"):
        for j in intent["joins"]:
            slots["joins"] += f" JOIN {j['table']} ON {j['on']}"

    if intent.get("filter_groups"):
        group_clauses = []
        for group in intent["filter_groups"]:
            conds = []
            for f in group["conditions"]:
                col, op, val = f["column"], f["operator"], f["value"]

                if op.upper() == "LIKE":
                    conds.append(f"{col} LIKE '%{f['value']}%'")
                elif op.upper() == "IN":
                    if isinstance(val, list):
                        formatted_vals = ", ".join(
                            f"'{v}'" if isinstance(v, str) and not str(v).replace('.', '', 1).isdigit() else str(v)
                            for v in val
                        )
                    else:
                        formatted_vals = f"'{val}'" if isinstance(val, str) else str(val)
                    conds.append(f"{col} IN ({formatted_vals})")
                else:
                    if isinstance(val, str) and not val.replace('.', '', 1).isdigit():
                        val = f"'{val}'"
                    conds.append(f"{col} {op} {val}")
            group_logic = group.get("logic", "AND")
            joined = f" {group_logic} ".join(conds)
            group_clauses.append(f"({joined})" if len(conds) > 1 else joined)
        slots["where"] = " WHERE " + " AND ".join(group_clauses)

    if intent.get("having"):
        h = intent["having"]
        slots["having"] = f" HAVING {h['aggregate']}({h['column']}) {h['operator']} {h['value']}"

    if intent.get("order_by"):
        o = intent["order_by"]
        slots["order_by"] = f" ORDER BY {o['column']} {o['direction']}"

    if intent.get("limit"):
        slots["limit"] = f" LIMIT {intent['limit']}"

    sql = SQL_TEMPLATE.format(**slots)
    return sql.strip()


# NOT traced — deterministic keyword check
def sql_guardrail(sql):
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE"]
    sql_upper = sql.upper()
    for word in forbidden:
        if word in sql_upper:
            return "i can only answer questions based on the data i cant update or modify the data "
    if not sql_upper.strip().startswith("SELECT"):
        return "i can only run read only queries on this data"
    return None

def validate_sql_matches_spec(sql: str, intent: dict) -> list[str]:
    errors = []
    if intent.get("aggregate") == "RATE":
        if "SUM(CASE" not in sql.upper() or "COUNT(*)" not in sql.upper():
            errors.append(f"Spec is RATE but SQL doesn't contain a SUM(CASE...)/COUNT(*) ratio: {sql}")
    elif intent.get("aggregate") and intent["aggregate"].upper() not in sql.upper():
        errors.append(f"SQL is missing aggregate '{intent['aggregate']}' from the spec: {sql}")
    if intent.get("filter_groups") and "WHERE" not in sql.upper():
        errors.append(f"Spec has filter_groups but SQL has no WHERE clause: {sql}")
    return errors


@traceable(name="run_pipeline")
def run_pipeline(query: str):
    examples = get_few_shot_examples(query)
    retrieved_schema_cols = final_schema_ranking(query, top_k=5)
    print("retrieved columns:", retrieved_schema_cols)

    intent = classify_intent(query, retrieved_schema_cols, examples)
    print("intent:", intent)

    sql_command = fill_sql_template(intent)
    guardrail_message = sql_guardrail(sql_command)
    if guardrail_message:
        return None, None, guardrail_message
    sql_errors = validate_sql_matches_spec(sql_command, intent)
    if sql_errors:
        print("[warning] SQL/spec mismatch:", sql_errors)

    print("sql:", sql_command)
    return intent, sql_command, None,retrieved_schema_cols


def cabin_missing(rows, columns, max_notes=3):
    notes = []
    if "Cabin" not in columns:
        return ""
    total_rows = [row for row in rows if row.get("Cabin") and "predicted" in str(row.get("Cabin")).lower()]
    if not total_rows:
        return ""
    for row in total_rows[:max_notes]:
        cabin = row.get("Cabin")
        pid = row.get("PassengerId")
        notes.append(
            f"This passenger {pid} cabin is statistical/ml based result ,the actual value is not in records"
        )

        rem_rows = len(total_rows) - max_notes
        if rem_rows > 0:
            notes.append(f"Like wise {rem_rows} passengers records are not present and itsv replaced with statistical method which says that he/she might be in that deck")
    return "\n\n".join(notes)


@traceable(name="run_sql", run_type="tool")
def run_sql(sql: str):
    result = pd.read_sql(sql, conn)
    result = result.where(pd.notnull(result), other=None)
    rows = []
    for record in result.to_dict(orient="records"):
        clean = {}
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        rows.append(clean)
    return rows, list(result.columns)


def build_result_summary(rows: list, columns: list, max_rows: int = 5) -> str:
    if not rows:
        return "No rows returned."
    if len(rows) <= max_rows:
        return f"columns: {columns}, rows: {json.dumps(rows, default=str)}"
    return f"columns: {columns}, row_count: {len(rows)}, sample: {json.dumps(rows[:3], default=str)}"


@traceable(name="summarize_result", run_type="llm")
def summarize_result(question: str, sql: str, rows: list, columns: list) -> str:
    preview = rows[:5]
    response = groq_client.chat.completions.create(
        model=groq_small_model,
        messages=[{
            "role": "user",
            "content": f"""Question: {question}
SQL used: {sql}
Columns: {columns}
Total rows returned: {len(rows)}
Sample rows (up to 5): {json.dumps(preview, default=str)}

Write a short, plain-English 1-3 sentence answer to the question based on this data.
Do not mention SQL. Do not repeat the raw rows. Just answer naturally."""
        }],
        temperature=0,
        reasoning_effort="low"
    )
    return response.choices[0].message.content.strip()

#here suggestion for follow up questions
@traceable(name="follow_up_suggestion_llm_call")
def followup_suggestion(sql,query,schemas,rows,cols,n=3):
    top_res=rows[:5]
    schema_text="\n".join(f"-{schema[col]}"for col in schemas)
    prompt=f"""you are a data analyst  working in the titanic dataset . user asked a question and 
        now u need to provide follow up question suggestion . so suggest {n} number of short , natural question related
        to the previous query and answer 
        Rules
        -keep the query under 12 words
        -Only suggest things answerable using these columns (never invent columns)
        - Make them useful: a drill-down, a comparison, or a related breakdown
        - Don't repeat the original question
        - Return ONLY a JSON array of strings, nothing else
        Available schemas:
        {schema_text}
        Generated sql:
        {sql}
        previous query:
        {query}
        sample rows:
        {json.dumps(top_res,default=str)}
        follow up questions (json array)"""
    response=groq_client.chat.completions.create(
        model=groq_small_model,
        messages=[{
            "role":"user",
            "content":prompt
        }],
        temperature=0.3)
    raw_ans=response.choices[0].message.content.strip()
    raw_ans = raw_ans.replace("```json", "").replace("```", "").strip()
    try:
        follow=json.loads(raw_ans)
        if isinstance(follow,dict):
            follow = follow.get("followups", [])
        return follow[:n]
    except json.JSONDecodeError:
        print(f"[warning] could not parse followups: {raw_ans}")
        return []

# step 7 — FastAPI app

class Turn(BaseModel):
    question: str
    sql: Optional[str] = None
    tables: Optional[list[str]] = None
    result_summary: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[Turn]] = []


app = FastAPI(title="Titanic SQL Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Titanic SQL Assistant is running"}


# This inner function is the ONE top-level traceable per request — everything
# it calls (standalone_question, run_pipeline, run_sql, summarize_result, etc.)
# nests underneath it automatically in the LangSmith dashboard as a single tree.
@traceable(name="process_query")
def process_query(question: str, history: list[dict]):
    if input_level_guardrail(question):
        description = "I can only answer questions about the data — I can't update, delete, or modify records."
        history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
        return {
            "success": True,
            "description": description,
         
            "rows": [],
            "history": history
        }
    standalone_query = standalone_question(question, history)
    category = classify_query(standalone_query)

    if category == "general":
        resp = groq_client.chat.completions.create(
            model=groq_small_model,
            messages=[{"role": "user", "content": f"""you are a titanic dataset expert and you need to explain about the questions related to the titanic dataset such as questions related to the columns present
                       or anything bounded inside the titanic dataset .
                       using the information provided  in 2-3 lines
                       schema description- {schema}
                       If the question is unrelated to the Titanic or this dataset, or too vague/ambiguous
                        to answer (e.g. missing context, unclear pronouns like "they"/"it" with nothing to
                        resolve them), respond with EXACTLY this and nothing else:
                        "I can only answer questions about the Titanic passengers in this dataset — things like age, sex, fare, class, survival, and family aboard. Could you rephrase your question around that?"
                        Otherwise, answer briefly and naturally.
                        Question: {question}
 
                       """}],
                       temperature=0.2,
                       reasoning_effort="low"
        )
        description = resp.choices[0].message.content.strip()
        history.append({"question": question,"tables": [], "result_summary": description})
        return {
            "success": True,
            "description": description,
         
            "rows": [],
            "history": history
        }

    try:
        intent, sql_command, guardrail2, retrieved_schemas = run_pipeline(standalone_query)
    except RateLimitError:
        raise  # let the outer handler in query_endpoint show the real rate-limit message
    except Exception as e:
        print(f"[error] run_pipeline failed: {e}")
        description = "I couldn't quite figure out how to answer that — could you rephrase it more simply?"
        history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
        return {
            "success": True,
            "description": description,
            "sql": None,
            "rows": [],
            "history": history
        }
    if guardrail2:
        history.append({"question": question, "sql": None, "tables": [], "result_summary": guardrail2})
        return {
            "success": True,
            "description": guardrail2,
         
            "rows": [],
            "history": history
        }

    rows, columns = run_sql(sql_command)
    description = summarize_result(question, sql_command, rows, columns)
    missed_value_msg = cabin_missing(rows, columns)
    if missed_value_msg:
        description = description + "\n\n" + missed_value_msg
    suggestion_ques=followup_suggestion(sql_command,standalone_query,retrieved_schemas,rows,columns)

    history.append({
        "question": question,
        "sql": sql_command,
        "tables": [intent.get("table")] if intent.get("table") else [],
        "result_summary": build_result_summary(rows, columns)
    })
    return {
        "success": True,
        "description": description,
        "sql": sql_command,
        "rows": rows,
        "suggested_ques":suggestion_ques,
        "history": history
        
    }

@app.post("/query-stream")
async def query_stream_endpoint(payload: QueryRequest):
    question = payload.question
    history = [msg.model_dump() for msg in payload.history] if payload.history else []

    def event_generator():
        try:
            if input_level_guardrail(question):
                description = "I can only answer questions about the data — I can't update, delete, or modify records."
                history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
                yield json.dumps({"type": "chunk", "text": description}) + "\n"
                yield json.dumps({"type": "final", "success": True, "description": description, "sql": None, "rows": [], "history": history}) + "\n"
                return

            standalone_query = standalone_question(question, history)
            category = classify_query(standalone_query)

            if category == "general":
                full_text = ""
                messages = [{"role": "user", "content": f"""you are a titanic dataset expert and you need to explain about the questions related to the titanic dataset such as questions related to the columns present
                       or anything bounded inside the titanic dataset .
                       using the information provided  in 2-3 lines
                       schema description- {schema}
                       If the question is unrelated to the Titanic or this dataset, or too vague/ambiguous
                        to answer (e.g. missing context, unclear pronouns like "they"/"it" with nothing to
                        resolve them), respond with EXACTLY this and nothing else:
                        "I can only answer questions about the Titanic passengers in this dataset — things like age, sex, fare, class, survival, and family aboard. Could you rephrase your question around that?"
                        Otherwise, answer briefly and naturally.
                        Question: {question}"""}]
                for piece in stream_groq_text(messages, temperature=0.2, reasoning_effort="low"):
                    full_text += piece
                    yield json.dumps({"type": "chunk", "text": piece}) + "\n"
                history.append({"question": question, "tables": [], "result_summary": full_text})
                yield json.dumps({"type": "final", "success": True, "description": full_text, "sql": None, "rows": [], "history": history}) + "\n"
                return

            try:
                intent, sql_command, guardrail2, retrieved_schemas = run_pipeline(standalone_query)
            except RateLimitError:
                raise
            except Exception as e:
                print(f"[error] run_pipeline failed: {e}")
                description = "I couldn't quite figure out how to answer that — could you rephrase it more simply?"
                history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
                yield json.dumps({"type": "chunk", "text": description}) + "\n"
                yield json.dumps({"type": "final", "success": True, "description": description, "sql": None, "rows": [], "history": history}) + "\n"
                return

            if guardrail2:
                history.append({"question": question, "sql": None, "tables": [], "result_summary": guardrail2})
                yield json.dumps({"type": "chunk", "text": guardrail2}) + "\n"
                yield json.dumps({"type": "final", "success": True, "description": guardrail2, "sql": None, "rows": [], "history": history}) + "\n"
                return

            rows, columns = run_sql(sql_command)

            preview = rows[:5]
            summary_messages = [{
                "role": "user",
                "content": f"""Question: {question}
SQL used: {sql_command}
Columns: {columns}
Total rows returned: {len(rows)}
Sample rows (up to 5): {json.dumps(preview, default=str)}

Write a short, plain-English 1-3 sentence answer to the question based on this data.
Do not mention SQL. Do not repeat the raw rows. Just answer naturally."""
            }]
            full_description = ""
            for piece in stream_groq_text(summary_messages, temperature=0, reasoning_effort="low"):
                full_description += piece
                yield json.dumps({"type": "chunk", "text": piece}) + "\n"

            missed_value_msg = cabin_missing(rows, columns)
            if missed_value_msg:
                full_description = full_description + "\n\n" + missed_value_msg
                yield json.dumps({"type": "chunk", "text": "\n\n" + missed_value_msg}) + "\n"

            suggestion_ques = followup_suggestion(sql_command, standalone_query, retrieved_schemas, rows, columns)

            history.append({
                "question": question,
                "sql": sql_command,
                "tables": [intent.get("table")] if intent.get("table") else [],
                "result_summary": build_result_summary(rows, columns)
            })
            yield json.dumps({
                "type": "final",
                "success": True,
                "description": full_description,
                "sql": sql_command,
                "rows": rows,
                "suggested_ques": suggestion_ques,
                "history": history
            }) + "\n"

        except RateLimitError as e:
            print(f"[rate limit] {e}")
            description = "I've hit my usage limit for the model right now. Please try again in a few minutes."
            history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
            yield json.dumps({"type": "chunk", "text": description}) + "\n"
            yield json.dumps({"type": "final", "success": True, "description": description, "sql": None, "rows": [], "history": history}) + "\n"
        except Exception as e:
            import traceback
            print(f"[error] /query-stream failed: {e}")
            print(traceback.format_exc())
            description = "I had trouble understanding that question. Could you try rephrasing it?"
            history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
            yield json.dumps({"type": "chunk", "text": description}) + "\n"
            yield json.dumps({"type": "final", "success": True, "description": description, "sql": None, "rows": [], "history": history}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
@app.post("/query")
def query_endpoint(payload: QueryRequest):
    question = payload.question
    history = [msg.model_dump() for msg in payload.history] if payload.history else []

    try:
        with trace(name="query_endpoint", metadata={"question": question}) as run:
            result = process_query(question, history)
            run.add_metadata({"success": result.get("success", True)})
            return result
    except RateLimitError as e:
        print(f"[rate limit] {e}")
        description = "I've hit my usage limit for the model right now. Please try again in a few minutes."
        history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
        return {
            "success": True,
            "description": description,
            "sql": None,
            "rows": [],
            "history": history
        }
    except Exception as e:
        import traceback
        print(f"[error] /query failed: {e}")
        print(traceback.format_exc())
        description = "I had trouble understanding that question. Could you try rephrasing it?"
        history.append({"question": question, "sql": None, "tables": [], "result_summary": description})
        return {
            "success": True,
            "description": description,
            "sql": None,
            "rows": [],
            "history": history
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)