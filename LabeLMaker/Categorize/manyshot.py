import tiktoken
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neighbors import NearestCentroid

from app.v01.schemas import CategorizationRequest
from LabeLMaker.Categorize.categorizer import BaseCategorizer
from LabeLMaker.utils.normalize_text import normalize_text
from LabeLMaker_config.config import Config


class ManyshotClassifier(BaseCategorizer):
    def __init__(self, categorization_request: CategorizationRequest, min_class_count: int):
        print("Initializing Many-shot categorizer")
        super().__init__()
        self.categorization_request = categorization_request
        self.min_class_count = min_class_count
        self.model = None
        self.client = Config.EMBEDDING_CLIENT
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.max_context_length = 8192

    def _get_filename(self):
        raise NotImplementedError

    def _get_mime_type(self):
        raise NotImplementedError

    def get_embeddings(self, text):
        """
        This function takes a text input, encodes it, truncates it if necessary, and then uses an Azure
        OpenAI client to embed the truncated text.

        Args:
          text: The `get_embeddings` function takes a text input as a parameter. This text input is then
        encoded using the `self.encoder.encode` method. The encoded tokens are then truncated based on the
        `max_context_length` and decoded back into text. If the truncated text is empty after stripping, the
        function

        Returns:
          The `get_embeddings` method returns the response from the Azure OpenAI client after encoding and
        querying the input text.
        """
        tokens = self.encoder.encode(text)
        truncated_text = self.encoder.decode(tokens[: self.max_context_length])
        if not truncated_text.strip():
            return None
        # Replace with your Azure OpenAI client code
        response = self.client.embed_query(truncated_text)
        return response

    def preprocess_data(self):
        """
        The `preprocess_data` function normalizes text data in both unlabeled and labeled examples for
        categorization.
        """
        # Unlabeled texts: these come from the full list.
        self.normalized_unlabeled_list = [
            normalize_text(text) if isinstance(text, str) else text
            for text in self.categorization_request.text_to_label
        ]

        # Normalize the training examples.
        self.normalized_example_list = []
        for example in self.categorization_request.examples:
            # Handle Example objects
            if hasattr(example, "text_with_label"):
                text_with_label = example.text_with_label
            # Allow tuple or list (text, label) format
            elif isinstance(example, (tuple, list)) and len(example) == 2:
                text_with_label = example[0]
            else:
                raise TypeError(
                    "Invalid example format. Expected Example object or tuple (text, label)."
                )

            self.normalized_example_list.append(
                normalize_text(text_with_label)
                if isinstance(text_with_label, str)
                else text_with_label
            )

    def select_model(self):
        """
        The `select_model` function chooses between Nearest Centroid and Multinomial Logistic Regression
        models based on a minimum class count threshold.
        """
        if self.min_class_count < Config.MIN_LOGISTIC_SAMPLES_PER_CLASS:
            self.model = NearestCentroid()
            self.model_name = "Nearest Centroid"
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                penalty="elasticnet",
                solver="saga",
                n_jobs=-1,
                l1_ratio=0.5,
                multi_class="multinomial",
            )
            self.model_name = "Multinomial Logistic Regression"

    def embed_data(self, texts):
        """
        The `embed_data` function takes a list of texts, retrieves embeddings for each text using the
        `get_embeddings` method, and returns a list of non-None embeddings.

        Args:
          texts: The `embed_data` method takes a list of texts as input. It then iterates over each text in
        the list, retrieves its embedding using the `get_embeddings` method, and appends the embedding to a
        list called `embeddings`. Finally, it returns the list of embeddings.

        Returns:
          The `embed_data` method returns a list of embeddings for the input texts.
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embeddings(text)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    def train_model(self):
        """
        The `train_model` function selects a model, extracts labels from examples, trains embeddings, and
        fits the model with the embeddings and labels.
        """
        self.select_model()
        labels = []
        for example in self.categorization_request.examples:
            # Handle Example objects
            if hasattr(example, "label"):
                labels.append(example.label)
            # Handle tuple or list (text, label)
            elif isinstance(example, (tuple, list)) and len(example) == 2:
                labels.append(example[1])
            else:
                raise TypeError(
                    "Invalid example format. Expected Example object or tuple (text, label)."
                )

        self.train_embeddings = self.embed_data(self.normalized_example_list)
        self.model.fit(self.train_embeddings, labels)

    def predict_unlabeled(self):
        """
        The `predict_unlabeled` function predicts labels for unlabeled text data and provides prediction
        probabilities and rationales for each prediction.

        Returns:
          The `predict_unlabeled` method returns a list of 4-tuples, where each tuple contains the unique
        ID, text, predicted category, and rationale for the prediction of unlabeled data points.
        """
        categorized_results = []
        unlabeled_text_embeddings = self.embed_data(self.normalized_unlabeled_list)
        unlabeled_labels = self.model.predict(unlabeled_text_embeddings)
        # Get prediction probabilities for each class for the unlabeled data.
        prediction_probabilities = self.model.predict_proba(unlabeled_text_embeddings)
        rationales = []
        for prob in prediction_probabilities:
            formatted_probs = [
                f"{cls}: {p:.4f}" if abs(p) >= 1e-4 else f"{cls}: {p:.4e}"
                for cls, p in zip(self.model.classes_, prob)
            ]
            rationale = " ".join(formatted_probs)
            rationales.append(rationale)

        # Build a 4-tuple for each prediction.
        # Assuming self.categorization_request.unique_ids exists.
        for uid, text, category, reason in zip(
            self.categorization_request.unique_ids,
            self.categorization_request.text_to_label,
            unlabeled_labels,
            rationales,
        ):
            categorized_results.append((uid, text, category, reason))
        return categorized_results

    def process(self):
        """
        Process the request by preprocessing data, training the model,
        and predicting labels for unlabeled text.
        Returns a list of 4-tuples (uid, text, predicted label, rationale).
        """
        st.write(str(self.__class__.__name__) + " operation in progress. Please wait...")
        self.preprocess_data()
        self.train_model()
        return self.predict_unlabeled()
