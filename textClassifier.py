from transformers import pipeline
import torch

#create zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

def classify(text, candidateLabels=["violent", "non-violent"], hypothesisTemplate="This text describes a real-life {} situation."):     #classify text into violent or non-violent
    result = classifier(
        sequences=text,
        candidate_labels=candidateLabels,
        hypothesis_template=hypothesisTemplate
    )
    #if confidence is low, return "unknown"
    if result["scores"][0] < 0.7:
        return {
            "label": "unknown",
            "score": 0.0,
            "all_scores": {}
        }
    
    label = result["labels"][0]
    score = result["scores"][0]
    allScores = dict(zip(result["labels"], result["scores"]))       #create a dictionary of all scores for each label

    return {
        "label": label,
        "score": score,
        "all_scores": allScores
    }