# Clinical NER with Bio_ClinicalBERT on i2b2 Dataset

This project fine-tunes **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) on the **i2b2 clinical notes dataset** for Named Entity Recognition (NER).  
Entities extracted include **PROBLEM**, **TREATMENT**, and **TEST**.

---

## Setup Instructions

### 1. Clone Repository & Install Requirements
```bash
git clone <your-repo-url>
cd <your-repo-name>

pip install -r requirements.txt
```

### 2. Requirements
- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.30
- Datasets >= 2.10
- scikit-learn
- pandas

Install using:
```bash
pip install torch transformers datasets scikit-learn pandas
```

---

## Dataset Preparation (i2b2)

1. Obtain the i2b2 dataset (requires license).  
2. Preprocess the dataset:
   - Replace blank lines with `_custom_note_separator_`.
   - Split into tokens and labels.  

---

## Training

```python
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
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

loaded_model = AutoModelForTokenClassification.from_pretrained("../results/checkpoint-best")
loaded_tokenizer = AutoTokenizer.from_pretrained("../results/checkpoint-best")

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

## Justification for Bio_ClinicalBERT

- Pre-trained on **MIMIC-III clinical notes** (real-world hospital data).  
- Outperforms general BERT in **clinical NLP tasks**.  
- Handles domain-specific terminology (medications, tests, symptoms).  
- Widely used in **biomedical research**.

---

## Error Analysis

**Probable Reason of error -**

- This i2b2 dataset often labels neurological exams like "diminished light touch" and "pinprick" as tests (they are exam findings).

- But semantically, they also look like problems/symptoms.

**Solution that might help -**
- Data Augumentation of the error cases and re-training

---

## Authors

- [@Sarvesh](https://github.com/Sarvesh326)
