# Demo Steps

## Prompt Chain

## Prompt 1

Hi, please update your knowledge base with the PDF patient reports and audio recordings that were uploaded. When finished, please list all files that were stored.

Please tell me all tests, test results, dates, and notes in the patient reports

Thank you! Please summarize the results of all patient data and audio recordings. Include a breakdown of the normal levels and abnormal levels. Include trends in reported levels across time, and display the data in an organized and intuitive way. After that, please use your medical training and web access to provide your expert medical analysis and insights on the data. Wherever possible, please include references to specific sources, including specific files in your knowledge base.

Please explain the results of all patient lab data. Include a breakdown of the normal levels and abnormal levels. Include trends in reported levels across time, and display the data in an organized and intuitive way. Please ensure every lab parameter extracted from the uploaded documents is individually analyzed. Do not summarize multiple panels into one — create distinct tables and analysis sections for each file's lab results.

After that, please use your medical training and web access to provide your expert medical analysis and insights on the data. Wherever possible, please include references to specific sources, including specific files in your knowledge base.

## Prompt 2

Thank you. Could you please add the below patient health profile to your memory?

Patient profile:

Age: 32 years
Gender: Female
Ethnicity: Indian
Previous medical history:

- anemia
- iron deficient
- Vitamin D deficient
- periods consisting of heavy flow
- appendicitis in December 2024
  - ruptured appendix
  - underwent appendectomy with heavy antibiotics
  - was septic temporarily
- probable GI issues / gastritis

Family medical history:

- mother had Stage 4 metastatic breast cancer that became terminal and developed into LMD
- maternal grandmother had rheumatoid artheritis
- paternal grandparents had type 2 diabetes

Patient's current symptoms:

- lump under left armpit
- Prickly/painful sensation in left breast - itchy

Below are some additional notes taken in January 2025. Many of these questions are unanswered, so please consider that as you compile a plan for us.

```
  PCP

    talked to her about the lump under left armpit - it went away after 1 month

    Prickly/painful sensation in left breast
      Is there redness, warmth, swelling?
      persistent lump, skin changes (dimpling, redness, or thickening), nipple discharge, or inversion?
      Skin looks dimpled like an orange peel?
      Skin changes around the nipple? (scaling, crusting, or redness)
      Pain radiating to the shoulder or arm? There was intermittent pain in shoulder and arm
      Headaches, vision changes?

    recommended BRCA test (gene)
      if she has that gene, higher chance for breast cancer

    she got a referral to dermatologist


  Gynecologist
    said she needs testing on the lump
    asked about extended panel of genetic testing

  Mammogram for left breast
    she got that ~jan 19
    negative - needs another eval

  Ultrasound on left breast
    done with mammogram by PCP
    negative - needs another eval by OBGYN

  Needs Genetic testing

  ultrasound for uterus
       negative

```

### Response 2

update_memory

Running:

- update_memory(task=...)

The patient's health profile has been successfully added to my memory. Here is a summary of the information stored:

### Patient Profile

- **Age**: 32 years
- **Gender**: Female
- **Ethnicity**: Indian

### Previous Medical History

- Anemia
- Iron deficiency
- Vitamin D deficiency
- Heavy menstrual flow
- Appendicitis in December 2024 (Ruptured appendix, underwent appendectomy with heavy antibiotics, was temporarily septic)
- Probable gastrointestinal issues / gastritis

### Family Medical History

- Mother had Stage 4 metastatic breast cancer, developed into leptomeningeal disease (LMD)
- Maternal grandmother had rheumatoid arthritis
- Paternal grandparents had type 2 diabetes

### Current Symptoms

- Lump under left armpit
- Prickly/painful sensation in left breast, including itching

### Notes (January 2025)

- PCP discussed lump under left armpit, which resolved after 1 month.
- Symptoms in the left breast, with several queries on associated symptoms.
- Recommended BRCA gene test; referral to a dermatologist.
- Gynecologist recommended testing on the lump and mentioned genetic testing.
- Mammogram and ultrasound conducted, both negative, but require further evaluation.
- Ultrasound for the uterus was negative.
- Genetic testing needed.

If you need further analysis or have additional questions, please let me know!

## Prompt 4

Please summarize the results of all patient data and audio recordings. Include a breakdown of the normal levels and abnormal levels. Include trends in reported levels across time, and display the data in an organized and intuitive way. After that, please use your medical training and PubMed access to provide your expert medical analysis and insights on the data. Wherever possible, please include references to specific sources, including specific files in your knowledge base.

# Prompt 5 - Upload PDF and ask about it

Could you please store all tests, test results, dates, and notes in this patient report? Please give me a detailed breakdown of each lab test, what the results mean, and any next steps and potential medications to consider.

# Alternate prompts

Please store all tests, test results, dates, and notes in this patient report.

# Response 1 - openbiollm-8b

Here's a detailed breakdown of the lab tests from the patient's Comprehensive Metabolic Panel:
