# Clinical NER with Bio_ClinicalBERT on i2b2 Dataset

This project fine-tunes **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) on the **i2b2 clinical notes dataset** for Named Entity Recognition (NER).  
Entities extracted include **PROBLEM**, **TREATMENT**, and **TEST**.

---
## Justification for Bio_ClinicalBERT

- Pre-trained on **MIMIC-III clinical notes** (real-world hospital data).  
- Outperforms general BERT in **clinical NLP tasks**.  
- Handles domain-specific terminology (medications, tests, symptoms).  
- Widely used in **biomedical research**.

---
## Setup Instructions

### 1. Clone Repository & Install Requirements
```bash
git clone https://github.com/Sarvesh326/NER_i2b2_Sarvesh.git
cd NER_i2b2_Sarvesh

pip install -r requirements.txt
```

### 2. Requirements
```txt
Python >= 3.8
PyTorch >= 1.10
Transformers >= 4.30
Datasets >= 2.10
scikit-learn
pandas
seqeval
```
### 3. Training Notebook Invalid

I have trained the model on a collab notebook since my local only supports CPU memory. For some unknown reason my `training.ipynb` notebook is not getting previewed in Github. 

To view the notebook please proceed using either of the following methods :

- Clone repo as mentioned above (pt1)
- Refer the collab link : [LINK](https://drive.google.com/file/d/1JBFmyCDMXC-VnR4dkfuVH1hFbe2tmFOm/view?usp=sharing)

---

## Dataset Preparation (i2b2)

1. Used the i2b2 dataset.  
2. Preprocessing:
   - Replace blank lines with `_custom_note_separator_`. This helps in segregating individual notes.
   - Split on `_custom_note_separator_` and then further split into tokens and labels.  

---

## Training

```python
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

training_args = TrainingArguments(
    output_dir="../models",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
```

---

## Evaluation

Evaluate model on validation set:
```python
# Run prediction on validation dataset
predictions_output = trainer.predict(tokenized_dataset["validation"])

true_labels, true_predictions = get_true_label_true_preds(predictions_output)

report = classification_report(true_labels, true_predictions)
print(report)
```

For **error analysis**:
```python
df_errors = build_error_dataframe(predictions_output, tokenized_dataset["validation"], id2label, tokenizer)
df_errors.head()
```

---

## Inference

Load trained model and run inference on **user input**:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

loaded_model = AutoModelForTokenClassification.from_pretrained("../models/checkpoint-xxxx")
loaded_tokenizer = AutoTokenizer.from_pretrained("../models/checkpoint-xxxx")

note = "Patient was prescribed aspirin for chest pain."
results, entities = predict_entities(note)

print("Problems:", entities["PROBLEM"])
print("Treatments:", entities["TREATMENT"])
print("Tests:", entities["TEST"])
```

Output:
```
Problems: ['chest pain']
Treatments: ['aspirin']
Tests: []
```
---

## Error Analysis

- This i2b2 dataset often labels neurological exams like "diminished light touch" and "pinprick" as tests (they are exam findings).
- But semantically, they also look like problems/symptoms.
- Due to this model hallucinates sometimes `TEST` as a `PROBLEM`.

**Solution that might help -**
- Data Augumentation of the error cases and re-training.

---

## Authors

- [@Sarvesh](https://github.com/Sarvesh326)
