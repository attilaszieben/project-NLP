import spacy

# Load the small English model you installed earlier
nlp = spacy.load("en_core_web_sm")

def preprocess_reviews(reviews):
    clean_reviews = []
    
    # nlp.pipe is much faster than a standard for-loop for large datasets
    for doc in nlp.pipe(reviews, batch_size=500):
        clean_tokens = []
        for token in doc:
            # Filter out stop words, punctuation, and white spaces
            if not token.is_stop and not token.is_punct and not token.is_space:
                # Apply lemmatization to get the base form and convert to lowercase
                clean_tokens.append(token.lemma_.lower())
                
        # Rejoin the clean tokens into a single string for the model
        clean_reviews.append(" ".join(clean_tokens))
        
    return clean_reviews

# Pass the raw text arrays you created in the previous step
print("Processing training data...")
clean_train_texts = preprocess_reviews(train_texts)

print("Processing test data...")
clean_test_texts = preprocess_reviews(test_texts)