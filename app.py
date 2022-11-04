import multiprocessing
import os
import unicodedata
from typing import List, Union

import numpy as np
import spacy
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Small spanish model from spacy
nlp = spacy.load("es_core_news_sm")
nlp.disable_pipe("parser")
nlp.add_pipe("sentencizer")

def get_sentences(
    speech_dir: str = './data/DiscursosOriginales',
    use_spacy: bool = False
):
    """Read each speech and return sentences.
    it uses nltk sentence tokenizer by default, otherwise
    It uses spacy sentence tokenizer - which may lead to better results but takes longer.
    """
    speech_files = os.listdir(speech_dir)

    # Iterate over all files in the DiscursosOriginales
    progress_bar = tqdm(speech_files)
    sentences = []
    for speech_file in progress_bar:
        with open(os.path.join(speech_dir, speech_file), 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            # Normalize text (remove strange characters)
            lines = [unicodedata.normalize("NFKD", line) for line in lines]

        # For each line in the speach use spacy's sentence tokenizer
        # and return sentences within speech lines
        for line in lines:
            if use_spacy:
                # Use spacy if desired
                doc = nlp(line)
                tokenized_sentences = [sent.text for sent in doc.sents]
            else:
                # Use NLTK otherwise
                tokenized_sentences = sent_tokenize(line, 'spanish')

            for sentence in tokenized_sentences:
                # Remove spaces or whitelines here
                if not sentence.isspace():
                    sentences.append(sentence)

    return sentences

def get_speech_sentences(
    speech_fn: str,
    use_spacy: bool = False
):
    """Takes a single speech file and returns its sentences.
    """
    with open(speech_fn, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        # Normalize text (remove strange characters)
        lines = [unicodedata.normalize("NFKD", line) for line in lines]

    progress_bar = tqdm(lines)
    sentences = []
    for line in progress_bar:
        if use_spacy:
            # Use spacy tokenizer
            doc = nlp(line)
            tokenized_sentences = [sent.text for sent in doc.sents]
        else:
            # Use NLTK otherwise
            tokenized_sentences = sent_tokenize(line, 'spanish')

        for sentence in tokenized_sentences:
            # Remove spaces or whitelines here
            if not sentence.isspace():
                sentences.append(sentence)

    return sentences

def get_sentences_tokens(sentences: List[str]):
    """Take a list of sentences and returns the tokenized sentence.
    """
    return [word_tokenize(sentence, "spanish") for sentence in sentences]

def get_sentence_tokens(sentence: str):
    """Take a single sentence and return a list of tokens.
    """
    return word_tokenize(sentence)

def train_w2v_model(
    sentences: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    epochs: int = 200,
    model_fname: Union[str, None] = None
):
    """Train a model and save it to disk if a file name is provided.
    """
    # Get all cores available
    cores = multiprocessing.cpu_count()
    # Train a model
    model = Word2Vec(
        sentences = sentences,
        vector_size = vector_size,
        window = window,
        workers = cores,
        epochs = epochs
    )
    # If a model name is provided then save it to disk
    if model_fname:
        model.save(model_fname)

    return model

def load_w2v_model(model_fname: str):
    """Simple wrapper to load a w2v model and return it and the vocabulary
    """
    # Load a model from the model's file name
    model = Word2Vec.load(model_fname)
    # get the models vocabulary
    vocab = model.wv.index_to_key

    return model, vocab


def get_sentence_embedding(
    model: Word2Vec,
    sentence: str
):
    sentence_vectors = []
    # Tokenize the full sentence
    words = get_sentence_tokens(sentence)
    for word in words:
        try:
            # Try getting the vectors of each word if exists
            vector = model.wv[word]
            sentence_vectors.append(vector)
        except KeyError:
            continue
    # Get the sentence embedding by calculating the mean of all
    # word embeddings in the sentence
    word_embeddings = np.array(sentence_vectors)
    if len(word_embeddings) > 0:
        sentence_embedding = word_embeddings.mean(axis = 0)
    else:
        sentence_embedding = np.zeros(shape = model.vector_size)

    return sentence_embedding

def get_document_embedding(
    model: Word2Vec,
    sentences: List[str]
):
    sentence_embeddings = np.array([get_sentence_embedding(model, sentence) for sentence in sentences])
    if len(sentence_embeddings) > 0:
        document_embedding = sentence_embeddings.mean(axis = 0)
    else:
        document_embedding = np.zeros(shape = model.vector_size)

    return document_embedding


def sort_sentences(
    model: Word2Vec,
    sentences: List[str],
):
    """The score of each sentence is calculated as the cosine_similarity
    between the sentence vector and the document vector (centroid of document).
    """
    document_embedding = get_document_embedding(model, sentences)
    scores = []
    for idx, sentence in enumerate(sentences):
        sentence_embedding = get_sentence_embedding(model, sentence)
        # Cosine similarity between document and sentence, i.e. how similar is
        # the sentence vector to the centroid of the document.
        score = get_embedding_similarity(
            v1 = sentence_embedding,
            v2 = document_embedding
        )
        scores.append((idx, score, sentence))

    scores.sort(key=lambda s: s[1], reverse=True)

    return scores


def get_embedding_similarity(
    v1: np.ndarray,
    v2: np.ndarray
) -> np.float32:
    """get the cosine similarity between two vectors (v1, v2)
    """

    score = cosine_similarity(
            v1.reshape(1, -1),
            v2.reshape(1, -1)
    )
    return score[0][0]


def create_summary(
    model: Word2Vec,
    sentences: List[str],
    n_summary_sentences: int,
    similarity_threshold: float,
    pretty_print: bool = True,
):
    """
    Implementation of the algorithm to summarize speeches for
    the class PROCESAMIENTO DE LENGUAJE NATURAL.
    - Cesar Flores
    - Fernando Garcia
    - Juan Francisco Pinto
    """
    summary = []
    scores = sort_sentences(model = model, sentences = sentences)
    for score in scores:
        if len(summary) >= n_summary_sentences:
            break
        s_idx, s_score, sent = score
        sentence_embedding = get_sentence_embedding(model = model, sentence = sent)
        add_sentence = False
        # There first sentence should always be added to the summary
        # so we check that if the summary doesn't contain sentences
        # then sentence 1 in order should be added by setting the value
        # `add_sentence` to True
        if len(summary) == 0:
            add_sentence = True

        # We check iteratively if the next sentence makes sense with the
        # rest of the sentences in the summary by looking at their
        # similarity and if they cross the threshold given or not.
        for summary_idx in range(len(summary)):
            summary_embedding = get_sentence_embedding(
                model = model,
                sentence = summary[summary_idx]
            )
            similarity = get_embedding_similarity(
                v1 = sentence_embedding,
                v2 = summary_embedding
            )
            if similarity >= similarity_threshold:
                # This is the only condition needed as the for loop
                # will not add repeated sentences to the summary.
                # `O[i] no incluida en Resumen`` this is not needed.
                add_sentence = True
            else:
                add_sentence = False

        # Finally we add the sentence to the summary list and continue
        # with the loop until we fill out all the conditions.
        if add_sentence:
            # print(f'''
            #       Sentence with index {s_idx} and
            #       importance score {s_score} added
            #       to the summary
            #     '''
            # )
            summary.append(sent)

    # Finally print a nice formatted summary of the speech
    # and return it's value as a string.
    formatted_summary = '\n\n'.join(summary)

    if pretty_print:
        print(formatted_summary)

    return formatted_summary
