"""
envclaims_prepare.py — replication data prep for Stammbach et al. (2023).

1. Downloads climatebert/environmental_claims and saves each split to data/envclaims/.
2. Checks whether the Table 4 / Table 5 guideline sentences are in the dataset
   (exact match after normalisation, else the closest sentence by similarity).
3. Writes seed candidates per rule from the TEST split only (prefilter for your reading;
   you pick the seeds by hand).
4. Writes data/envclaims/texts_test.csv (id, text) for generate_variants.py --texts.

    pip install datasets
    python scripts\\envclaims_prepare.py
"""
import difflib
import os
import re

import pandas as pd
from datasets import load_dataset

OUT = "data/envclaims"
os.makedirs(OUT, exist_ok=True)

# ---- 1. dataset -------------------------------------------------------------
ds = load_dataset("climatebert/environmental_claims")
frames = []
for split in ds:
    d = ds[split].to_pandas()
    d["split"] = split
    d["id"] = [f"{split}_{i:04d}" for i in range(len(d))]
    d.to_csv(f"{OUT}/{split}.csv", index=False, encoding="utf-8")
    frames.append(d)
    print(f"{split:<11} n={len(d):>5}  share label 1 = {d['label'].mean():.3f}")
alld = pd.concat(frames, ignore_index=True)


def norm(t):
    t = str(t).lower().replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r"[^a-z0-9%]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


alld["norm"] = alld["text"].map(norm)
lookup = dict(zip(alld["norm"], zip(alld["id"], alld["split"], alld["label"])))

# ---- 2. guideline sentences (Tables 4-5, transcribed from the paper) ---------
GUIDE = [
    ("T4", 1, "Farmers who operate under this scheme are required to dedicate 10% of their land to wildlife preservation."),
    ("T4", 1, "We prove our commitment to a sustainable world every day—by being a force for change where we work and live and holding ourselves and our suppliers to high standards in three vital aspects of doing business: people, product, and planet."),
    ("T4", 1, "Our places, which are designed to meet high sustainability standards, become part of local communities, provide opportunities for skills development and employment and promote wellbeing."),
    ("T4", 1, "Fast Retailing has adopted \"Unlocking the Power of Clothing\" for its Sustainability Statement, and through the apparel business seeks to contribute to the sustainable development of society."),
    ("T4", 1, "Hermès, which is currently managed by the sixth generation of family shareholders, is aware of its social responsibility and strives to give back to the world a part of what it gives to the Company."),
    ("T4", 1, "In 2016, UTC was placed on the CDP climate change and supplier A List, and in 2017 and 2018 received an A- and Leadership designation."),
    ("T4", 1, "Change internal behavior; Drive low-carbon investment; Identify and seize low-carbon opportunities; Stakeholder expectations."),
    ("T4", 1, "We are looking into the Insurance Underwriting element, and have taken part in the CRO Forum's Sustainability Carbon Footprinting paper of Underwriting."),
    ("T4", 1, "In a further demonstration of the importance we place on helping customers to live sustainably, we became signatories of the Task Force on Climate related Financial Disclosures, to provide consistent information to our stakeholders."),
    ("T4", 1, "As for assets, DBJ Green Building certification for 18 properties, BELS certification for 33 properties, and CASBEE certification for one property have been received."),
    ("T4", 1, "Our clean, safe and high-tech products and solutions enable everything from food production to space travel, improving the everyday life of people everywhere."),
    ("T4", 1, "FreshPoint, our specialty produce company, addresses customers' needs for fresh, unique, organic, and local produce items."),
    ("T4", 1, "WilLDAR consists of detecting methane leaks with an optical gas imaging camera and repairing those leaks within 30 days."),
    ("T4", 1, "These products include climate metrics, Climate Value-at-Risk (VAR), carbon portfolio reporting, low carbon, and climate change indexes as well as tools to identify clean-tech and environmentally oriented companies."),
    ("T5", 0, "We do this for 15 sustainable and impact strategies (equities, bonds and green bonds)."),
    ("T5", 0, "We use the EcoAct ClimFIT (Climate Financial Institutions Tool) tool to measure the carbon emissions associated with the household and personal products sector."),
    ("T5", 0, "AUSEA is a miniaturized sensor, fitted onto a commercial drone, that can detect methane and carbon dioxide."),
    ("T5", 0, "This will further accelerate Croda's positive impact by creating and delivering solutions to tackle some of the biggest challenges the world is facing."),
    ("T5", 0, "Hence, the Scope 2 emission is included in the Scope 1 emission which has been reported in accordance with the ISO 14064-1 requirements as verified by qualified independent assessor."),
    ("T5", 0, "Emissions associated with processing activities are associated with the supply of these ingredients and are included in our Scope 3 supply chain emissions."),
    ("T5", 0, "Emissions are modelled based on sector averages including linear regression and country carbon emissions intensities for GDP."),
    ("T5", 0, "Wood products facilities also operate lumber drying kilns and other processes that can either use the steam from the boilers or, if direct fired, will commonly use natural gas."),
    ("T5", 0, "We use the EcoAct ClimFIT (Climate Financial Institutions Tool) tool to measure the carbon emissions associated with utilities."),
    ("T5", 0, "In the past we have conducted analysis of our portfolio impact on the climate, using scope 3 as a metric."),
    ("T5", 0, "For that, Danone needs organic fresh milk."),
    ("T5", 0, "UPM Biofuels is developing a new feedstock concept by growing Brassica Carinata as a sequential crop in South America."),
    ("T5", 0, "Our key sources of emissions are the running of our operations (electricity, business travel, etc), purchased goods and services (consultants, maintenance work, IT services, etc), and land leased to sheep and beef farming (to keep the grass low under our wind farms)."),
    ("T5", 0, "Extreme weather events and the impacts of transitioning to a low-carbon economy have the potential to disrupt business activities, damage property, and otherwise affect the value of assets, and affect our customers' ability to repay loans."),
    ("T5", 0, "At the date of this report, the Group owns 34 mills (29 of which produce containerboard), 245 converting plants (most of which convert containerboard into corrugated boxes), 40 recovered fibre facilities and two wood procurement operations (which together provide raw material for our mills) and 34 other production facilities carrying on other related activities."),
]

rows = []
norms = alld["norm"].tolist()
for table, glabel, sent in GUIDE:
    n = norm(sent)
    if n in lookup:
        did, split, lab = lookup[n]
        rows.append([table, glabel, "exact", 1.0, did, split, lab, sent[:70]])
        continue
    best = difflib.get_close_matches(n, norms, n=1, cutoff=0.0)
    ratio = difflib.SequenceMatcher(a=n, b=best[0]).ratio() if best else 0.0
    if best and ratio >= 0.85:
        did, split, lab = lookup[best[0]]
        rows.append([table, glabel, "near", round(ratio, 3), did, split, lab, sent[:70]])
    else:
        rows.append([table, glabel, "absent", round(ratio, 3), "", "", "", sent[:70]])

g = pd.DataFrame(rows, columns=["table", "paper_label", "match", "similarity", "dataset_id",
                                "split", "dataset_label", "sentence_start"])
g.to_csv(f"{OUT}/guideline_sentences_in_dataset.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 60)
print("\nGuideline sentences found in the dataset:")
print(g.drop(columns="sentence_start").to_string(index=False))
print(g["match"].value_counts().to_string())
lab_conflict = g[(g["match"] != "absent") & (g["dataset_label"].astype(str) != g["paper_label"].astype(str))]
if len(lab_conflict):
    print(f"\nNOTE: {len(lab_conflict)} guideline sentence(s) carry a different label in the dataset.")

# ---- 3. seed candidates from TEST split -------------------------------------
test = alld[alld["split"] == "test"].copy()
guide_ids = set(g["dataset_id"]) - {""}
test = test[~test["id"].isin(guide_ids)]
t = test["text"].str.lower()

PRODUCT = r"\b(product|products|solution|solutions|service|services|technology|technologies|process|processes|system|systems|sensor|facility|facilities|plant|plants)\b"
IMPACT = r"\b(reduc\w*|lower\w*|cut\w*|avoid\w*|sav\w*|efficien\w*|cleaner|less (?:carbon|energy|water|waste)|low[- ]carbon|emission[- ]free|renewable)\b"
SUSTAIN = r"\b(sustainab\w*|responsib\w*)\b"
ENVWORD = r"\b(environment\w*|climate|carbon|emission\w*|energy|green|renewable|ecolog\w*|biodivers\w*|waste|water|pollut\w*)\b"
RISK = r"\b(risk\w*|exposure|exposed|source\w* of emissions|physical|extreme weather|transition)\b"
COMMIT = r"\b(commit\w*|aim\w*|target\w*|we will|strive\w*|working to|plan to|pledge\w*|ambition)\b"

cand = []
for _, r in test.iterrows():
    s = r["text"].lower()
    has = lambda p: re.search(p, s) is not None
    if has(PRODUCT) and r["label"] == 0 and not has(IMPACT):
        cand.append((r["id"], "stated_impact", r["label"], r["text"]))
    if has(PRODUCT) and r["label"] == 1 and has(IMPACT) and not has(SUSTAIN) and not has(COMMIT):
        cand.append((r["id"], "stated_impact", r["label"], r["text"]))
    if r["label"] == 1 and has(SUSTAIN) and not has(r"\b(climate|carbon|emission\w*|renewable|biodivers\w*|pollut\w*)\b"):
        cand.append((r["id"], "sustainability_wording", r["label"], r["text"]))
    if r["label"] == 0 and not has(SUSTAIN) and not has(ENVWORD):
        cand.append((r["id"], "sustainability_wording", r["label"], r["text"]))
    if r["label"] == 0 and has(RISK) and not has(COMMIT):
        cand.append((r["id"], "commitment_to_act", r["label"], r["text"]))
    if r["label"] == 1 and has(RISK) and has(COMMIT):
        cand.append((r["id"], "commitment_to_act", r["label"], r["text"]))

c = pd.DataFrame(cand, columns=["id", "rule", "label", "text"])
c["wc"] = c["text"].str.split().str.len()
c = c[(c["wc"] >= 8) & (c["wc"] <= 45)].sort_values(["rule", "label", "wc"])
c.to_csv(f"{OUT}/seed_candidates_envclaims.csv", index=False, encoding="utf-8")
print("\nSeed candidates (test split, 8-45 words) by rule and label:")
print(pd.crosstab(c["rule"], c["label"]).to_string())

# ---- 4. texts file for the generator ----------------------------------------
alld[["id", "text"]].to_csv(f"{OUT}/texts_all.csv", index=False, encoding="utf-8")
print(f"\nwrote {OUT}/: split CSVs, guideline_sentences_in_dataset.csv, "
      "seed_candidates_envclaims.csv, texts_all.csv")
