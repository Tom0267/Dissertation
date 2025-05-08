from collections import defaultdict
import math
import re

def getViolenceScore(responseText, returnExplanation):
    response = responseText.lower().strip()
    totalWeight = 0.0
    matchedWeights = defaultdict(float)

    #define category weights
    categories = {
        "violent": {
            "fight": 1.0,
            "fighting": 1.0,
            "punch": 0.9,
            "kick": 0.9,
            "aggression": 0.85,
            "violent": 1.0,
            "physical altercation": 0.95,
            "attack": 0.95,
            "conflict": 0.7,
            "hitting": 0.85,
            "struggle": 0.7,
            "aggressive": 0.8
        },
        "nonviolent": {
            "no violence": -1.0,
            "not violent": -0.95,
            "non-violent": -0.95,
            "peaceful": -0.9,
            "calm": -0.8,
            "no fighting": -0.95,
            "quiet scene": -0.8,
            "nothing aggressive": -0.8
        },
        "uncertain": {
            "maybe": -0.5,
            "unclear": -0.6,
            "not sure": -0.6,
            "hard to tell": -0.6,
            "ambiguous": -0.5
        },
        "affirmation": {
            r"\byes\b": 0.6,    #regex pattern to match whole word "yes"
            r"\bno\b": -0.6     #regex pattern to match whole word "no"
        }
    }

    #apply regex or plain matching based on category
    for group, terms in categories.items():
        for term, weight in terms.items():
            if term.startswith(r"\b"):  #regex pattern
                if re.search(term, response):
                    matchedWeights[group] += weight
                    totalWeight += weight
            else:
                if term in response:
                    matchedWeights[group] += weight
                    totalWeight += weight

    violenceProb = 1 / (1 + math.exp(-totalWeight)) #sigmoid function to convert totalWeight to probability 

    if returnExplanation:
        return round(violenceProb, 3), {
            "score": round(violenceProb, 3),
            "matchedTerms": dict(matchedWeights),
            "rawScore": round(totalWeight, 3),
            "input": responseText
        }

    return round(violenceProb, 3)