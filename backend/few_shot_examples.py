# ── Few-shot examples ─────────────────────────────────────────────────────────
# Written as one big string block (easy to read/edit), then parsed below into
# a list of {"question": ..., "sql": ...} dicts, which is the form main.py
# needs for embedding + indexed lookup (FEW_SHOT_EXAMPLES[int(idx)]).

_FEW_SHOT_TEXT = """
Question: What is the death rate in each passenger class?
SQL:
SELECT 
    Pclass,
    COUNT(*) AS total_passengers,
    SUM(CASE WHEN Survived = 0 THEN 1 ELSE 0 END) AS total_deaths,
    ROUND(SUM(CASE WHEN Survived = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS death_rate_pct
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question:Give me complete details of female passengers over 60 who survived
SQL:
SELECT * FROM titanic WHERE (Sex = 'female' AND Age > 60 AND Survived = 1)
---
Question: What is the survival rate for each class?
SQL:
SELECT 
    Pclass,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: What is the survival rate for male vs female passengers?
SQL:
SELECT 
    Sex,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
GROUP BY Sex
---
Question: Which embarked port had the worst survival rate?
SQL:
SELECT 
    Embarked,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
WHERE Embarked IS NOT NULL
GROUP BY Embarked
ORDER BY survival_rate_pct ASC
---
Question: What is the death rate for each class broken down by gender?
SQL:
SELECT 
    Pclass,
    Sex,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS death_rate_pct
FROM titanic
GROUP BY Pclass, Sex
ORDER BY Pclass, Sex
---
Question: What is the survival rate by age group and class?
SQL:
SELECT 
    Age_group,
    Pclass,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
GROUP BY Age_group, Pclass
ORDER BY Age_group, Pclass
---
Question: How many passengers were in each class?
SQL:
SELECT 
    Pclass,
    COUNT(*) AS total_passengers
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: How many passengers embarked from each port?
SQL:
SELECT 
    Embarked,
    COUNT(*) AS total_passengers
FROM titanic
WHERE Embarked IS NOT NULL
GROUP BY Embarked
---
Question: How many passengers are there in total?
SQL:
SELECT COUNT(*) AS total_passengers FROM titanic
---
Question: How many male passengers survived?
SQL:
SELECT COUNT(*) AS survivor_count FROM titanic WHERE Sex = 'male' AND Survived = 1
---
Question: How many passengers embarked from either Cherbourg or Queenstown?
SQL:
SELECT COUNT(*) AS passenger_count FROM titanic WHERE Embarked = 'C' OR Embarked = 'Q'
---
Question: What is the average fare paid by each class?
SQL:
SELECT 
    Pclass,
    ROUND(AVG(Fare), 2) AS avg_fare
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: What is the oldest age recorded in the dataset?
SQL:
SELECT MAX(Age) AS oldest_age FROM titanic
---
Question: What is the maximum fare paid by any passenger?
SQL:
SELECT MAX(Fare) AS max_fare FROM titanic
---
Question: What is the average age of survivors versus non-survivors?
SQL:
SELECT 
    Survived,
    ROUND(AVG(Age), 2) AS avg_age
FROM titanic
WHERE Age IS NOT NULL
GROUP BY Survived
---
Question: Who are the top 5 passengers who paid the highest fare?
SQL:
SELECT 
    PassengerId, Name, Fare
FROM titanic
ORDER BY Fare DESC
LIMIT 5
---
Question: Who was the youngest passenger on board?
SQL:
SELECT 
    PassengerId, Name, Age
FROM titanic
WHERE Age IS NOT NULL
ORDER BY Age ASC
LIMIT 1
---
Question: Which class had the most survivors?
SQL:
SELECT 
    Pclass,
    SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) AS survivor_count
FROM titanic
GROUP BY Pclass
ORDER BY survivor_count DESC
LIMIT 1
---
---
Question: Who paid the highest fare?
SQL:
SELECT *
FROM titanic
ORDER BY Fare DESC
LIMIT 1
---
Question: Who was the oldest passenger on board?
SQL:
SELECT *
FROM titanic
ORDER BY Age DESC
LIMIT 1
---
Question: Who was the cheapest ticket sold to?
SQL:
SELECT *
FROM titanic
ORDER BY Fare ASC
LIMIT 1
---
Question: What is the highest fare paid?
SQL:
SELECT MAX(Fare) FROM titanic
---
Question: Show me passengers who paid more than 100 fare and survived?
SQL:
SELECT 
    PassengerId, Name, Sex, Age, Pclass, Fare
FROM titanic
WHERE Fare > 100 AND Survived = 1
ORDER BY Fare DESC
---
Question: List all female passengers in first class who did not survive.
SQL:
SELECT 
    PassengerId, Name, Age, Fare
FROM titanic
WHERE Sex = 'female' AND Pclass = 1 AND Survived = 0
---
Question: What are the different embarkation ports in the dataset?
SQL:
SELECT DISTINCT Embarked FROM titanic WHERE Embarked IS NOT NULL
---
Question: Give me their name and cabin
SQL:
SELECT Name, Cabin
FROM titanic
WHERE Survived = 1
---
Question: What age groups exist in this dataset?
SQL:
SELECT DISTINCT Age_group FROM titanic
---
Question: Which classes had more than 100 survivors?
SQL:
SELECT 
    Pclass,
    SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) AS survivor_count
FROM titanic
GROUP BY Pclass
HAVING SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) > 100
"""


def _parse_examples(raw: str) -> list[dict]:
    examples = []
    for block in raw.strip().split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        question = block.split("\n")[0].replace("Question:", "").strip()
        sql = block.split("SQL:", 1)[1].strip()
        examples.append({"question": question, "sql": sql})
    return examples


FEW_SHOT_EXAMPLES = _parse_examples(_FEW_SHOT_TEXT)