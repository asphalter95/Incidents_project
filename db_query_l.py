import pyodbc
import pandas as pd
from datetime import datetime


conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=tcp:hqitsmdbs3.avp.ru,1433;"
    "DATABASE=ARSystem;"
    "Trusted_Connection=yes;"
)

query = r"""
WITH ExpertsIncidents AS (
    SELECT DISTINCT incident_number
    FROM HPD_Worklog
    WHERE 
        submit_date BETWEEN 
            DATEDIFF(SECOND, '1970-01-01 03:00:00.000', DATEADD(day, -7, GETDATE()))
        AND DATEDIFF(SECOND, '1970-01-01 03:00:00.000', GETDATE())
        AND detailed_description LIKE '%->%%SD EXPERTS%'
        AND incident_number IN (
            SELECT DISTINCT Incident_Number
            FROM KSP_HPD_Search_Worklog_Join_Re AS WW
            WHERE WW.Submit_Date_WL BETWEEN 
                    DATEDIFF(SECOND, '1970-01-01 03:00:00.000', DATEADD(day, -7, GETDATE()))
                AND DATEDIFF(SECOND, '1970-01-01 03:00:00.000', GETDATE())
        )
)

SELECT  
    DATEADD(SECOND, W.Submit_Date, '1970-01-01 03:00:00.000') AS [Worklog Date],
    HD.Description AS [Incident Title],
    HD.Detailed_Decription AS [Incident Notes],
    W.Incident_Number,
    W.Detailed_Description,

    CASE   
        WHEN W.Work_Log_Type = '2000'  THEN 'Customer Communication'
        WHEN W.Work_Log_Type = '8000'  THEN 'General Information'
        WHEN W.Work_Log_Type = '16000' THEN 'Email System'
        WHEN W.Work_Log_Type = '20000' THEN 'Reassignment'
        WHEN W.Work_Log_Type = '21001' THEN 'Assignee Changed'
        WHEN W.Work_Log_Type = '21002' THEN 'Owner Changed'
        WHEN W.Work_Log_Type = '21003' THEN 'Status Changed'
        WHEN W.Work_Log_Type = '21004' THEN 'StatusReason Changed'
        ELSE 'NOT Defined'
    END AS Work_Log_Type,

    W.Work_Log_Type,

    CASE   
        WHEN W.Work_Log_Date > (
            SELECT TOP 1 W1.Work_Log_Date   
            FROM HPD_WorkLog AS W1
            WHERE W1.Incident_Number = W.Incident_Number
              AND W1.Description LIKE '%-> [[]SD FIRST%'
            ORDER BY W1.Work_Log_Date ASC
        )
        THEN '>'
        ELSE '<='
    END AS [More than]

FROM HPD_WorkLog W WITH (NOLOCK)
JOIN ExpertsIncidents EI 
    ON EI.incident_number = W.Incident_Number
JOIN HPD_Help_Desk HD WITH (NOLOCK)
    ON HD.Incident_Number = W.Incident_Number

ORDER BY W.Incident_Number, W.Submit_Date;
"""
ts = datetime.now().strftime("%Y%m%d")
output_path = fr"week_{ts}.csv"

with pyodbc.connect(conn_str) as conn:
    df = pd.read_sql_query(query, conn)

df.to_csv(output_path, index=False, sep=';')
print(f"Saved {len(df)} rows to {output_path}")

import re
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import catboost
import datetime
import os
import platform
import smtplib
import requests
import subprocess
from email.mime.text import MIMEText

def send_mail(subject: str, body: str, receivers: list, attachments: list=None):

    receivers_ps = f"-To " + ", ".join([f'"{r}"' for r in receivers])
    attach_block =""
    if attachments:
        attach_block = f"-Attachments " + ", ".join([f'"{a}"' for a in attachments])

    message = f'''
$body = @"
{body}
"@;

Send-MailMessage -From "RECEIVER" {receivers_ps} -Subject "{subject}" -Body $body -Encoding UTF8 -SmtpServer "SMTPSERVER" {attach_block}
'''
    subprocess.run(['powershell', '-Command', message], check=True)
    #print("[OK] Email sent successfully!")

# BASE_MODEL_PATH = "./ruroberta_tickets_mlm_50000"
# TOKENIZER_PATH = "text_tokenizer"
CATBOOST_PATH = "catboost_light.cbm"

# DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# class TextOnlyClassifier(nn.Module):
#     def __init__(self, model_name, hidden=64):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.encoder.requires_grad_(False)

#         h = self.encoder.config.hidden_size
#         self.classifier = nn.Sequential(
#             nn.Linear(h, h // 2),
#             nn.ReLU(),
#             nn.Linear(h // 2, 2)
#         )

#     def forward(self, input_ids, attention_mask):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         mask = attention_mask.unsqueeze(-1)
#         emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
#         logits = self.classifier(emb)
#         return logits


# class ProdDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_len=512):
#         self.texts = list(texts)
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.texts[idx],
#             max_length=self.max_len,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#         }



def load_models():
    # # токенайзер
    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # # текстовая голова BERT
    # text_model = TextOnlyClassifier(BASE_MODEL_PATH).to(DEVICE)
    # state_dict = torch.load("text_only_head.pt", map_location=DEVICE)
    # text_model.load_state_dict(state_dict, strict=False)
    # text_model.eval()

    # CatBoost
    cat = catboost.CatBoostClassifier()
    cat.load_model(CATBOOST_PATH)

    return cat

#   PREPROCESS

def preprocess_for_bert(text: str) -> str:
    text = str(text)
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    text = re.sub(r'\S+@\S+', ' <EMAIL> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим выгрузку Remedy к формату табличных фич + текст,
    как в твоём пайплайне.
    Ожидается, что в df_raw есть:
    - 'Worklog Date'
    - 'Incident Title'
    - 'Incident Notes'
    - 'Incident_Number'
    - 'Detailed_Description'
    - 'Work_Log_Type.1'
    """
    df = df_raw.copy()
    df['Worklog Date'] = pd.to_datetime(df['Worklog Date'])
    df['Detailed_Description'] = df['Detailed_Description'].astype(str) + '. '

    df = (
        df
        .groupby(['Incident Title', 'Incident Notes', 'Incident_Number'])
        .agg(
            text=('Detailed_Description', 'sum'),
            Log_Count=('Worklog Date', 'count'),
            Avg_Log_Interval=('Worklog Date', lambda x: x.sort_values().diff().dt.total_seconds().mean() / 3600),
            Max_Log_Interval=('Worklog Date', lambda x: x.sort_values().diff().dt.total_seconds().max() / 3600),
            Status_Change_Count=('Work_Log_Type.1', lambda x: (x == 21003).sum()),
            Reassignments=('Work_Log_Type.1', lambda x: (x == 20000).sum()),
            Customer_Communications=('Work_Log_Type.1', lambda x: (x == 2000).sum()),
            Was_Reopen=('Work_Log_Type.1', lambda x: int((x == 21005).any())),
            Wrong_Assignment=(
                'Detailed_Description',
                lambda s: int(
                    s.astype(str)
                    .str.contains('Wrong Assignment', case=False, na=False)
                    .any()
                )
            ),
            First_Date=('Worklog Date', 'min'),
            Last_Date=('Worklog Date', 'max')
        )
        .reset_index()
    )

    df['Lifetime_hours'] = (df['Last_Date'] - df['First_Date']).dt.total_seconds() / 3600

    df['text'] = df.apply(
        lambda x: 'Title ' + str(x['Incident Title']) +
                  '. Notes ' + str(x['Incident Notes']) +
                  '. Worklog ' + str(x['text']),
        axis=1
    )

    df.drop(['Incident Title', 'Incident Notes', 'First_Date', 'Last_Date'], axis=1, inplace=True)
    df['text'] = df['text'].apply(preprocess_for_bert)

    return df

# SCORING

# def predict_tickets(df: pd.DataFrame,
#                     tokenizer,
#                     text_model,
#                     cat_model,
#                     threshold: float | None = None):

#     df = df.copy()

#     # ---- TEXT MODEL ----
#     text_ds = ProdDataset(df["text"], tokenizer)
#     loader = DataLoader(text_ds, batch_size=32)

#     text_probs = []
#     with torch.no_grad():
#         for batch in loader:
#             logits = text_model(
#                 batch["input_ids"].to(DEVICE),
#                 batch["attention_mask"].to(DEVICE)
#             )
#             pr = torch.softmax(logits, dim=1)[:, 1]
#             text_probs.extend(pr.cpu().numpy())

#     df["p_bad_text"] = text_probs

#     CAT_FEATURES = [
#         "Log_Count",
#         "Avg_Log_Interval",
#         "Max_Log_Interval",
#         "Status_Change_Count",
#         "Reassignments",
#         "Customer_Communications",
#         "Lifetime_hours",
#         "p_bad_text"
#     ]

#     X = df[CAT_FEATURES].astype(float)
#     assert list(X.columns) == CAT_FEATURES

#     proba = cat_model.predict_proba(X)[:, 1]

#     df["score"] = proba

#     return df

def predict_tickets_light(df: pd.DataFrame, cat, threshold=None):

    CAT_FEATURES = [
        "Log_Count",
        "Avg_Log_Interval",
        "Max_Log_Interval",
        "Status_Change_Count",
        "Reassignments",
        "Customer_Communications",
        "Lifetime_hours"
    ]

    X = df[CAT_FEATURES].astype(float)
    assert list(X.columns) == CAT_FEATURES


    proba = cat.predict_proba(X)[:, 1]

    df["score"] = proba

    return df

# WEEKLY DIAGNOSTICS

def weekly_score_stats(df_scored: pd.DataFrame) -> dict:
    scores = df_scored["score"].values
    n = len(scores)

    stats = {}
    stats["n_tickets"] = int(n)
    stats["score_median"] = round(float(np.median(scores)), 4)
    stats["score_mean"] = round(float(np.mean(scores)), 4)
    stats["score_p95"] = round(float(np.quantile(scores, 0.95)), 4)
    stats["score_p99"] = round(float(np.quantile(scores, 0.99)), 4)

    k = int(n * 0.2)
    stats["top_20_size"] = k

    token_len = df_scored["text"].str.len()
    stats["text_len_mean"] = round(float(token_len.mean()), 2)
    stats["text_len_std"] = round(float(token_len.std()), 2)
    stats["text_len_p95"] = round(float(token_len.quantile(0.95)), 2)
    stats["text_len_p99"] = round(float(token_len.quantile(0.99)), 2)

    cols = [
        "Log_Count", "Avg_Log_Interval", "Max_Log_Interval",
        "Status_Change_Count", "Reassignments",
        "Customer_Communications", "Lifetime_hours"
    ]
    cols = [c for c in cols if c in df_scored.columns]

    if cols:
        tab_stats = df_scored[cols].agg(["mean", "std", "min", "max"])
        stats["tabular_stats"] = {
            col: {
                idx: round(float(tab_stats.loc[idx, col]), 2)
                for idx in tab_stats.index
            }
            for col in tab_stats.columns
        }
    else:
        stats["tabular_stats"] = {}
    with open(f'stats_{str(datetime.datetime.now().date())}', "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats

def weekly_score_df(stats: dict) -> pd.DataFrame:
    flat = {
        "n_tickets": stats["n_tickets"],
        "score_mean": stats["score_mean"],
        "score_median": stats["score_median"],
        "score_p95": stats["score_p95"],
        "score_p99": stats["score_p99"],
        "text_len_mean": stats["text_len_mean"],
        "text_len_p95": stats["text_len_p95"],
    }
    return pd.DataFrame([flat])

def weekly_score_file(stats: dict, date_file: str) -> pd.DataFrame:
    flat = {
        "date": date_file,
        "n_tickets": stats["n_tickets"],
        "score_mean": stats["score_mean"],
        "score_median": stats["score_median"],
        "score_p95": stats["score_p95"],
        "score_p99": stats["score_p99"],
        "text_len_mean": stats["text_len_mean"],
        "text_len_p95": stats["text_len_p95"],
    }
    return pd.DataFrame([flat])

def scores_to_df():
    import json
    folder = os.path.abspath('')
    dfs = []
    for filename in os.listdir(folder):
        if filename.startswith("stats_"):
            file_path = os.path.join(folder, filename)
            date_file = filename.replace("stats_", "").replace(".json", "")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = weekly_score_file(data, date_file)
            dfs.append(df)
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=final_df.iloc[:,1:].columns)
        final_df['date'] = pd.to_datetime(final_df['date'], errors = 'coerce')
        final_df = final_df.sort_values('date', ascending=True).reset_index(drop=True)
        delta = pd.DataFrame(((final_df.iloc[-1,1:].to_numpy()-final_df.iloc[0,1:].to_numpy())/final_df.iloc[0,1:].to_numpy()*100).reshape(1,-1), columns=final_df.columns[1:])
        delta.insert(0,'date', 'delta %')
        final_df = pd.concat([final_df, delta], ignore_index=True)
        print(final_df)
        return final_df
    else:
        print("Нет файлов stats_*.json в папке")
        return

cat_model = load_models()
df.columns.values[-2] = 'Work_Log_Type.1'
df_raw = df.copy()#pd.read_csv(f'week_{ts}.csv', sep=";")
df_pre = preprocess_data(df_raw)
df_pre = df_pre[df_pre['text'].str.contains('[Resolved]',regex=False) | df_pre['text'].str.contains('[Cancelled]',regex=False)]
df_scored = predict_tickets_light(df_pre, cat_model, threshold=None)
df_scored.sort_values("score", ascending=False).to_csv(f'week_ready_{ts}.csv', sep=';')
df_all = df_scored.sort_values("score", ascending=False)['Incident_Number']
df_all.to_csv('ALL.csv', encoding='utf-8-sig', index=False)
top20 = df_all.iloc[:int(len(df_scored)*0.2)]
top20.to_csv('TOP20.csv', encoding='utf-8-sig', index=False)
weekly_score_stats(df_scored)

send_mail(
    subject=f"Отчет INCIDENTS от {datetime.datetime.today().date()}",
    body=f"Больше информации на сервере SERVER в INCIDENTS1.ps1 на рабочем столе.\n\n\nСписок TOP20% подозрительных инцидентов на прошлой неделе:\n{', '.join(top20.to_list())}. \n\n\nСписок ВСЕХ на прошлой неделе, отсортировано по убыванию подозрительности:\n{', '.join(df_all.to_list())}",
    receivers=['RECEIVERLIST'],
    attachments = ['TOP20.csv', 'ALL.csv']
)