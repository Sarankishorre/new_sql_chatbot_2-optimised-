"""
Gold-standard evaluation set for the Titanic SQL Assistant.

Each case has a hand-written "gold_sql" — the SQL you consider definitely
correct for that question. The eval runner executes BOTH your pipeline's
generated SQL and the gold SQL against the same database, then compares the
actual returned rows (execution accuracy) — not just the SQL text. This is
the standard approach in text-to-SQL research because two different SQL
strings can return the identical correct answer.

"category" lets you see accuracy broken down by capability, not just one
blended score.
"""

EVAL_CASES = [

    # ---------- aggregate ----------
    {
        "question": "How many passengers survived?",
        "category": "aggregate",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Survived = 1",
    },
    {
        "question": "What is the average fare paid by passengers?",
        "category": "aggregate",
        "gold_sql": "SELECT AVG(Fare) FROM titanic",
    },
    {
        "question": "What is the maximum age among all passengers?",
        "category": "aggregate",
        "gold_sql": "SELECT MAX(Age) FROM titanic",
    },
    {
        "question": "What is the minimum fare paid?",
        "category": "aggregate",
        "gold_sql": "SELECT MIN(Fare) FROM titanic",
    },
    {
        "question": "What is the total sum of fares collected?",
        "category": "aggregate",
        "gold_sql": "SELECT SUM(Fare) FROM titanic",
    },
    {
        "question": "How many passengers were there in total?",
        "category": "aggregate",
        "gold_sql": "SELECT COUNT(*) FROM titanic",
    },

    # ---------- simple filter ----------
    {
        "question": "How many male passengers were there?",
        "category": "filter",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Sex = 'male'",
    },
    {
        "question": "How many passengers were above age 60?",
        "category": "filter",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Age > 60",
    },
    {
        "question": "How many passengers paid a fare under 10?",
        "category": "filter",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Fare < 10",
    },
    {
        "question": "How many passengers were in first class?",
        "category": "filter",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Pclass = 1",
    },

    # ---------- rate / percentage ----------
    {
        "question": "What was the survival rate for women?",
        "category": "rate",
        "gold_sql": "SELECT SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM titanic WHERE Sex = 'female'",
    },
    {
        "question": "What percentage of passengers survived?",
        "category": "rate",
        "gold_sql": "SELECT SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM titanic",
    },
    {
        "question": "What was the survival rate for passengers above age 20?",
        "category": "rate",
        "gold_sql": "SELECT SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM titanic WHERE Age > 20",
    },
    {
        "question": "What proportion of first class passengers survived?",
        "category": "rate",
        "gold_sql": "SELECT SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM titanic WHERE Pclass = 1",
    },

    # ---------- group by ----------
    {
        "question": "Average age of passengers by class",
        "category": "group_by",
        "gold_sql": "SELECT Pclass, AVG(Age) FROM titanic GROUP BY Pclass",
    },
    {
        "question": "Survival rate split by sex",
        "category": "group_by",
        "gold_sql": "SELECT Sex, SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM titanic GROUP BY Sex",
    },
    {
        "question": "Number of passengers in each class",
        "category": "group_by",
        "gold_sql": "SELECT Pclass, COUNT(*) FROM titanic GROUP BY Pclass",
    },
    {
        "question": "Average fare paid by each passenger class",
        "category": "group_by",
        "gold_sql": "SELECT Pclass, AVG(Fare) FROM titanic GROUP BY Pclass",
    },

    # ---------- order by / limit ----------
    {
        "question": "Show the top 5 oldest passengers",
        "category": "order_limit",
        "gold_sql": "SELECT * FROM titanic ORDER BY Age DESC LIMIT 5",
    },
    {
        "question": "List the 3 cheapest tickets sold",
        "category": "order_limit",
        "gold_sql": "SELECT * FROM titanic ORDER BY Fare ASC LIMIT 3",
    },

    # ---------- multi-condition ----------
    {
        "question": "How many female passengers in first class survived?",
        "category": "multi_condition",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Sex = 'female' AND Pclass = 1 AND Survived = 1",
    },
    {
        "question": "How many male passengers above age 30 did not survive?",
        "category": "multi_condition",
        "gold_sql": "SELECT COUNT(*) FROM titanic WHERE Sex = 'male' AND Age > 30 AND Survived = 0",
    },
]