"""Clinical story templates used by the synthetic data generator.

Everything here is fictional teaching content. No real patient data is used or implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Setting = Literal["outpatient", "inpatient", "emergency"]
Direction = Literal["high", "low", "normal"]
VitalsBias = Literal["normal", "febrile", "hypoxic"]


@dataclass(frozen=True)
class LabSpec:
    """A laboratory test: display name, spellings seen in the wild and a reference range."""

    key: str
    name: str
    aliases: tuple[str, ...]
    unit: str
    unit_variants: tuple[str, ...]
    low: float
    high: float
    decimals: int = 1


@dataclass(frozen=True)
class ImagingTemplate:
    title: str
    technique: str
    findings: str
    impression: str


@dataclass(frozen=True)
class Archetype:
    key: str
    setting: Setting
    chief: str
    hpi: tuple[str, ...]
    assessment: str
    plan: str
    diagnoses: tuple[tuple[str, str], ...]
    labs: tuple[tuple[str, Direction], ...]
    imaging: tuple[ImagingTemplate, ...]
    course: str
    meds: str
    vitals_bias: VitalsBias = "normal"


LAB_CATALOG: dict[str, LabSpec] = {
    spec.key: spec
    for spec in (
        LabSpec(
            "glucose",
            "Glucose",
            ("Glucose, Serum", "GLU", "Blood Glucose"),
            "mg/dL",
            ("mg/dL", "mg/dl", "MG/DL"),
            70,
            99,
            0,
        ),
        LabSpec(
            "hba1c",
            "Hemoglobin A1c",
            ("HbA1c", "A1C", "Glycated Hemoglobin"),
            "%",
            ("%", "percent"),
            4.0,
            5.6,
            1,
        ),
        LabSpec(
            "creatinine",
            "Creatinine",
            ("Creat", "Serum Creatinine"),
            "mg/dL",
            ("mg/dL", "mg/dl"),
            0.6,
            1.2,
            2,
        ),
        LabSpec(
            "egfr",
            "eGFR",
            ("GFR est", "Estimated GFR"),
            "mL/min/1.73m2",
            ("mL/min/1.73m2", "mL/min/1.73 m2"),
            90,
            120,
            0,
        ),
        LabSpec(
            "potassium",
            "Potassium",
            ("K", "Serum Potassium"),
            "mmol/L",
            ("mmol/L", "mmol/l", "mEq/L"),
            3.5,
            5.1,
            1,
        ),
        LabSpec(
            "sodium", "Sodium", ("Na", "Serum Sodium"), "mmol/L", ("mmol/L", "mEq/L"), 135, 145, 0
        ),
        LabSpec(
            "wbc",
            "White Blood Cell Count",
            ("WBC", "Leukocytes"),
            "10*3/uL",
            ("10*3/uL", "x10^3/uL", "K/uL"),
            4.5,
            11.0,
            1,
        ),
        LabSpec(
            "hemoglobin", "Hemoglobin", ("Hgb", "HGB"), "g/dL", ("g/dL", "g/dl"), 12.0, 16.0, 1
        ),
        LabSpec(
            "ferritin",
            "Ferritin",
            ("Ferritin, Serum",),
            "ng/mL",
            ("ng/mL", "ng/ml", "ug/L"),
            30,
            300,
            0,
        ),
        LabSpec(
            "bnp", "B-type Natriuretic Peptide", ("BNP",), "pg/mL", ("pg/mL", "pg/ml"), 0, 100, 0
        ),
        LabSpec(
            "crp",
            "C-Reactive Protein",
            ("CRP", "C Reactive Protein"),
            "mg/L",
            ("mg/L", "mg/l"),
            0,
            10,
            1,
        ),
        LabSpec(
            "tsh",
            "Thyroid Stimulating Hormone",
            ("TSH",),
            "mIU/L",
            ("mIU/L", "uIU/mL"),
            0.4,
            4.0,
            2,
        ),
    )
}

ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="pneumonia",
        setting="inpatient",
        chief="Cough, fever and shortness of breath",
        hpi=(
            "{age}-year-old {sex} with 4 days of productive cough, fever to 102 F and "
            "pleuritic right-sided chest pain.",
            "{age}-year-old {sex} with worsening dyspnea and fever for 3 days despite "
            "over-the-counter antipyretics.",
        ),
        assessment="Community-acquired pneumonia, right lower lobe (J18.9) with mild "
        "hypoxemia (R09.02).",
        plan="Start ceftriaxone and azithromycin. Supplemental oxygen to keep SpO2 above 92%. "
        "Repeat chest X-ray in 6 weeks to document resolution.",
        diagnoses=(("J18.9", "Community-acquired pneumonia"), ("R09.02", "Hypoxemia")),
        labs=(("wbc", "high"), ("crp", "high"), ("sodium", "normal"), ("creatinine", "normal")),
        imaging=(
            ImagingTemplate(
                "XR CHEST 2 VIEWS",
                "PA and lateral views of the chest.",
                "Right lower lobe airspace consolidation with air bronchograms. Small right "
                "parapneumonic effusion. Heart size is normal. No pneumothorax.",
                "Right lower lobe pneumonia with small parapneumonic effusion.",
            ),
        ),
        course="Treated with IV ceftriaxone and azithromycin, transitioned to oral therapy on "
        "hospital day 3. Oxygen weaned to room air. Afebrile for 48 hours before discharge.",
        meds="Amoxicillin-clavulanate 875 mg twice daily for 5 days; acetaminophen as needed.",
        vitals_bias="hypoxic",
    ),
    Archetype(
        key="diabetes",
        setting="outpatient",
        chief="Diabetes follow-up with elevated home glucose readings",
        hpi=(
            "{age}-year-old {sex} with type 2 diabetes reporting fasting home glucose 180-240 "
            "mg/dL, increased thirst and urination.",
            "{age}-year-old {sex} with long-standing type 2 diabetes, missed several metformin "
            "doses, home readings trending upward.",
        ),
        assessment="Type 2 diabetes mellitus with hyperglycemia (E11.65). HbA1c above goal.",
        plan="Increase metformin to 1000 mg twice daily, add empagliflozin 10 mg daily. "
        "Diabetic foot and eye examinations. Dietitian referral. Recheck HbA1c in 3 months.",
        diagnoses=(("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),),
        labs=(("hba1c", "high"), ("glucose", "high"), ("creatinine", "normal"), ("egfr", "normal")),
        imaging=(),
        course="Not applicable (outpatient management).",
        meds="Metformin 1000 mg twice daily; empagliflozin 10 mg daily.",
    ),
    Archetype(
        key="heart_failure",
        setting="inpatient",
        chief="Worsening leg swelling and difficulty breathing when lying flat",
        hpi=(
            "{age}-year-old {sex} with known heart failure presenting with 2 weeks of "
            "progressive leg edema, orthopnea and weight gain of 6 kg.",
            "{age}-year-old {sex} with ischemic cardiomyopathy and increasing exertional "
            "dyspnea, nocturnal cough and bilateral ankle swelling.",
        ),
        assessment="Acute on chronic systolic heart failure (I50.23), volume overloaded.",
        plan="IV furosemide with strict intake and output. Daily weights. Continue "
        "guideline-directed therapy. Cardiology consult and echocardiogram.",
        diagnoses=(("I50.23", "Acute on chronic systolic heart failure"),),
        labs=(("bnp", "high"), ("creatinine", "high"), ("sodium", "low"), ("potassium", "normal")),
        imaging=(
            ImagingTemplate(
                "XR CHEST 1 VIEW PORTABLE",
                "Portable AP view of the chest.",
                "Cardiomegaly with pulmonary vascular congestion and interstitial edema. "
                "Small bilateral pleural effusions.",
                "Findings consistent with congestive heart failure.",
            ),
            ImagingTemplate(
                "ECHOCARDIOGRAM TRANSTHORACIC",
                "Complete transthoracic echocardiogram with Doppler.",
                "Left ventricular ejection fraction approximately 30% with global hypokinesis. "
                "Moderate functional mitral regurgitation. Left atrium moderately dilated.",
                "Reduced left ventricular systolic function, LVEF 30%.",
            ),
        ),
        course="Diuresed 5 L with IV furosemide. Weight down 5 kg. Transitioned to oral "
        "torsemide. Education on sodium and fluid restriction.",
        meds="Torsemide 40 mg daily; carvedilol 12.5 mg twice daily; sacubitril-valsartan.",
    ),
    Archetype(
        key="ckd",
        setting="outpatient",
        chief="Nephrology follow-up for chronic kidney disease",
        hpi=(
            "{age}-year-old {sex} with chronic kidney disease and hypertension, no new "
            "complaints, mild fatigue.",
            "{age}-year-old {sex} with progressive decline in renal function over 18 months, "
            "occasional muscle cramps.",
        ),
        assessment="Chronic kidney disease stage 3b (N18.32) with hyperkalemia (E87.5).",
        plan="Dietary potassium restriction. Start patiromer. Avoid NSAIDs. Repeat renal "
        "panel in 4 weeks. Renal ultrasound.",
        diagnoses=(("N18.32", "Chronic kidney disease stage 3b"), ("E87.5", "Hyperkalemia")),
        labs=(
            ("creatinine", "high"),
            ("egfr", "low"),
            ("potassium", "high"),
            ("hemoglobin", "low"),
        ),
        imaging=(
            ImagingTemplate(
                "US RETROPERITONEUM RENAL",
                "Real-time grayscale and Doppler ultrasound of both kidneys.",
                "Both kidneys are echogenic with reduced cortical thickness. No hydronephrosis, "
                "calculi or focal mass.",
                "Chronic medical renal disease without obstruction.",
            ),
        ),
        course="Not applicable (outpatient management).",
        meds="Patiromer 8.4 g daily; amlodipine 5 mg daily.",
    ),
    Archetype(
        key="migraine",
        setting="emergency",
        chief="Severe headache with sensitivity to light",
        hpi=(
            "{age}-year-old {sex} with sudden onset throbbing right-sided headache, nausea and "
            "photophobia for 10 hours. History of similar episodes.",
            "{age}-year-old {sex} with recurrent headaches, today severe with vomiting and "
            "visual shimmering. No head trauma, no fever.",
        ),
        assessment="Migraine without aura (G43.009). No red-flag features on examination.",
        plan="IV fluids, metoclopramide and ketorolac with improvement. CT head to exclude "
        "acute pathology. Outpatient neurology follow-up.",
        diagnoses=(("G43.009", "Migraine without aura"),),
        labs=(("glucose", "normal"), ("wbc", "normal")),
        imaging=(
            ImagingTemplate(
                "CT HEAD WITHOUT CONTRAST",
                "Axial non-contrast CT of the head.",
                "No acute intracranial hemorrhage, mass effect or large territorial infarct. "
                "Ventricles and sulci are normal. Paranasal sinuses are clear.",
                "Negative non-contrast CT of the head.",
            ),
        ),
        course="Symptoms resolved in the emergency department. Discharged home.",
        meds="Sumatriptan 50 mg as needed; ondansetron 4 mg as needed.",
    ),
    Archetype(
        key="wrist_fracture",
        setting="emergency",
        chief="Left wrist pain after a fall",
        hpi=(
            "{age}-year-old {sex} slipped on ice and fell on an outstretched left hand. "
            "Immediate pain, swelling and deformity of the wrist.",
            "{age}-year-old {sex} fell from standing height onto left hand, unable to grip, "
            "wrist swollen and tender over the distal radius.",
        ),
        assessment="Closed distal radius fracture, left (S52.502A), neurovascularly intact.",
        plan="Reduction under hematoma block and sugar-tong splint. Analgesia. Orthopedic "
        "clinic follow-up in 1 week. Elevate and ice.",
        diagnoses=(("S52.502A", "Closed fracture of distal radius, left"),),
        labs=(),
        imaging=(
            ImagingTemplate(
                "XR WRIST LEFT 3 VIEWS",
                "PA, lateral and oblique views of the left wrist.",
                "Transverse fracture of the distal radial metaphysis with dorsal angulation "
                "and mild impaction. No intra-articular extension. Ulnar styloid intact.",
                "Acute distal radius fracture with dorsal angulation.",
            ),
        ),
        course="Closed reduction performed, post-reduction films acceptable alignment.",
        meds="Ibuprofen 600 mg every 8 hours as needed; acetaminophen.",
    ),
    Archetype(
        key="anemia",
        setting="outpatient",
        chief="Fatigue and shortness of breath on exertion",
        hpi=(
            "{age}-year-old {sex} with 3 months of fatigue, pallor and breathlessness climbing "
            "stairs. Heavy menstrual history noted.",
            "{age}-year-old {sex} with progressive tiredness, brittle nails and pica. No "
            "melena or hematochezia reported.",
        ),
        assessment="Iron deficiency anemia (D50.9). Source evaluation pending.",
        plan="Start oral ferrous sulfate 325 mg daily. Repeat CBC and ferritin in 8 weeks. "
        "Gastroenterology referral for endoscopic evaluation.",
        diagnoses=(("D50.9", "Iron deficiency anemia"),),
        labs=(("hemoglobin", "low"), ("ferritin", "low"), ("wbc", "normal")),
        imaging=(),
        course="Not applicable (outpatient management).",
        meds="Ferrous sulfate 325 mg daily with vitamin C.",
    ),
    Archetype(
        key="pyelonephritis",
        setting="inpatient",
        chief="Fever, flank pain and burning with urination",
        hpi=(
            "{age}-year-old {sex} with 2 days of dysuria, urinary frequency, fever to 103 F "
            "and right flank pain.",
            "{age}-year-old {sex} with chills, rigors and costovertebral angle tenderness "
            "after several days of urinary symptoms.",
        ),
        assessment="Acute pyelonephritis (N10) with mild acute kidney injury.",
        plan="IV ceftriaxone, aggressive hydration. Urine and blood cultures pending. "
        "CT abdomen and pelvis to exclude obstruction or abscess.",
        diagnoses=(("N10", "Acute pyelonephritis"),),
        labs=(("wbc", "high"), ("creatinine", "high"), ("crp", "high")),
        imaging=(
            ImagingTemplate(
                "CT ABDOMEN PELVIS WITH CONTRAST",
                "Axial CT of the abdomen and pelvis after IV contrast.",
                "Right kidney is enlarged with striated nephrogram and perinephric stranding. "
                "No obstructing calculus, hydronephrosis or drainable abscess.",
                "Findings consistent with acute right pyelonephritis.",
            ),
        ),
        course="Fever resolved within 48 hours on IV ceftriaxone. Cultures grew E. coli, "
        "sensitive to ciprofloxacin. Transitioned to oral therapy.",
        meds="Ciprofloxacin 500 mg twice daily for 7 days.",
        vitals_bias="febrile",
    ),
    Archetype(
        key="atrial_fibrillation",
        setting="outpatient",
        chief="Palpitations and irregular heartbeat",
        hpi=(
            "{age}-year-old {sex} with intermittent palpitations for 3 weeks, occasional "
            "lightheadedness, no syncope.",
            "{age}-year-old {sex} noticing an irregular pulse on a home monitor, mild "
            "exertional fatigue.",
        ),
        assessment="Atrial fibrillation (I48.91), CHA2DS2-VASc score 3, rate controlled.",
        plan="Start apixaban 5 mg twice daily. Continue metoprolol for rate control. "
        "Echocardiogram. Check thyroid function.",
        diagnoses=(("I48.91", "Unspecified atrial fibrillation"),),
        labs=(
            ("tsh", "normal"),
            ("potassium", "normal"),
            ("creatinine", "normal"),
            ("hemoglobin", "normal"),
        ),
        imaging=(
            ImagingTemplate(
                "ECHOCARDIOGRAM TRANSTHORACIC",
                "Complete transthoracic echocardiogram with Doppler.",
                "Left ventricular ejection fraction 55%. Left atrium is moderately enlarged. "
                "No significant valvular stenosis or regurgitation.",
                "Preserved LVEF with left atrial enlargement.",
            ),
        ),
        course="Not applicable (outpatient management).",
        meds="Apixaban 5 mg twice daily; metoprolol succinate 50 mg daily.",
    ),
    Archetype(
        key="copd",
        setting="inpatient",
        chief="Increased shortness of breath and wheezing",
        hpi=(
            "{age}-year-old {sex} with COPD and a 40 pack-year smoking history, 3 days of "
            "increased sputum, wheeze and dyspnea at rest.",
            "{age}-year-old {sex} with chronic obstructive pulmonary disease using rescue "
            "inhaler every 2 hours, worsening cough.",
        ),
        assessment="COPD with acute exacerbation (J44.1), no pneumonia on imaging.",
        plan="Nebulized albuterol-ipratropium, prednisone 40 mg for 5 days, doxycycline. "
        "Smoking cessation counseling. Pulmonary rehabilitation referral.",
        diagnoses=(("J44.1", "COPD with acute exacerbation"),),
        labs=(("wbc", "high"), ("glucose", "high"), ("sodium", "normal")),
        imaging=(
            ImagingTemplate(
                "XR CHEST 2 VIEWS",
                "PA and lateral views of the chest.",
                "Hyperinflated lungs with flattened diaphragms and increased retrosternal air "
                "space. No focal consolidation, effusion or pneumothorax.",
                "Hyperinflation consistent with COPD. No acute infiltrate.",
            ),
        ),
        course="Improved with bronchodilators and steroids, discharged on oral prednisone "
        "taper and home nebulizer.",
        meds="Prednisone 40 mg daily for 5 days; tiotropium inhaler; albuterol as needed.",
        vitals_bias="hypoxic",
    ),
    Archetype(
        key="hypertension",
        setting="outpatient",
        chief="Blood pressure follow-up and medication refill",
        hpi=(
            "{age}-year-old {sex} with essential hypertension, adherent to medications, home "
            "readings 130-140/80-90, no headaches or chest pain.",
            "{age}-year-old {sex} presenting for routine hypertension review, no side effects "
            "from lisinopril.",
        ),
        assessment="Essential hypertension (I10), near goal.",
        plan="Continue lisinopril 20 mg daily. Low-sodium diet, regular exercise. Annual "
        "metabolic panel. Return in 6 months.",
        diagnoses=(("I10", "Essential hypertension"),),
        labs=(("potassium", "normal"), ("creatinine", "normal"), ("glucose", "normal")),
        imaging=(),
        course="Not applicable (outpatient management).",
        meds="Lisinopril 20 mg daily.",
    ),
    Archetype(
        key="appendicitis",
        setting="inpatient",
        chief="Right lower quadrant abdominal pain",
        hpi=(
            "{age}-year-old {sex} with 24 hours of periumbilical pain migrating to the right "
            "lower quadrant, anorexia and nausea.",
            "{age}-year-old {sex} with progressive right lower quadrant pain, low-grade fever "
            "and rebound tenderness.",
        ),
        assessment="Acute appendicitis (K35.80), without perforation.",
        plan="NPO, IV fluids and antibiotics. Surgical consultation for laparoscopic appendectomy.",
        diagnoses=(("K35.80", "Acute appendicitis"),),
        labs=(("wbc", "high"), ("crp", "high")),
        imaging=(
            ImagingTemplate(
                "CT ABDOMEN PELVIS WITH CONTRAST",
                "Axial CT of the abdomen and pelvis after IV contrast.",
                "Dilated appendix measuring 11 mm with periappendiceal fat stranding and an "
                "appendicolith. No free air or drainable collection.",
                "Acute uncomplicated appendicitis.",
            ),
        ),
        course="Underwent uncomplicated laparoscopic appendectomy. Tolerating diet on day 1, "
        "discharged on day 2.",
        meds="Acetaminophen and ibuprofen as needed for pain.",
        vitals_bias="febrile",
    ),
)
