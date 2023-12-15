import spacy

def perform_sentence_segmentation(text):
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text using SpaCy
    doc = nlp(text)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    return sentences

# Example usage
input_text = "Right now you're at the entrance of a room. Now turn right, there is an open door in front of you, move towards the entrance of the room and enter into the room. It is your end point."

segmented_sentences = perform_sentence_segmentation(input_text)
#print(segmented_sentences)

# Print the segmented sentences
#for idx, sentence in enumerate(segmented_sentences):
 #   print(f"({idx + 1}) {sentence}")
