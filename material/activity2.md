# Lab:

> [!NOTE]  
> Throughout this lab, add text blocks to explain any new concepts you encounter. This will help reinforce your understanding and ensure clarity in your learning process. If you're unsure about a function or concept, consider writing a short definition/explanation in a text block. In addition, you can use your favorite LLM  to get more details.

----


## **Part 1: Introduction to Text Classification with Transformers and Gradio**

**Introduction**

1.  **Objective**:
    *   Introduce the concept of **supervised learning** through **text classification**.
    *   Demonstrate the use of the **DistilBERT** model for **sentiment analysis**.
    *   Utilize **Gradio** for creating a basic interactive interface.

2.  **Learning Outcomes**:
    *   Understand the procedure for using a pre-trained **DistilBERT** model for text classification.
    *   Learn to build a simple user interface for NLP model interaction using **Gradio**.

3.  **Tools**:
    *   Google Colab (requires a Google Account).
    *   Hugging Face `transformers` library.
    *   `gradio` library.

**Setup: Environment Preparation in Google Colab**

Open a new Google Colab notebook ([colab.research.google.com](https://colab.research.google.com)). Execute the following command in a code cell to install the required libraries:

```python
!pip install transformers gradio
```
This command installs the `transformers` and `gradio` libraries. Wait for the installation process to complete.

---

**Section 1.1: Console-based Sentiment Analysis**

1.  **Description**:
    *   **Objective**: Load a pre-trained sentiment analysis model and use it to predict the sentiment of provided text samples. Compare predictions against known labels.

2.  **Steps**:

    *   **Step 1: Import libraries:** Import the `pipeline` function from the `transformers` library.
    *   **Step 2: Load the pre-trained model:** Instantiate a `pipeline` for `"sentiment-analysis"` using the `"distilbert-base-uncased"` model. This model is pre-trained for language understanding tasks.
    *   **Step 3: Prepare sample data:** Define a list containing text strings, each associated with a known sentiment label (`"positive"` or `"negative"`).
    *   **Step 4: Perform classification and display results:** Iterate through the sample data. For each text sample, obtain the sentiment prediction from the loaded `classifier`. Print the original text, its true label, and the predicted label.

3.  **Code**: Execute the following code in a new Colab cell:

    ```python
    # Step 1: Import the pipeline function
    from transformers import pipeline

    # Step 2: Load the pre-trained sentiment analysis model
    print("Loading model...")
    # This initializes the sentiment analysis pipeline with the specified model.
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    print("Model loaded.")

    # Step 3: Sample data with corresponding labels
    data = [
        ("I love this product! It's amazing.", "positive"),
        ("This is the worst experience I've ever had.", "negative"),
        ("I feel great after using this service.", "positive"),
        ("The customer support was horrible and unhelpful.", "negative"),
        ("It's okay, not great but not terrible either.", "negative") # Neutral text often gets classified as positive/negative by binary classifiers
    ]

    print("\n--- Analyzing Sentiments ---")
    # Step 4: Classify sentiment for each sample and display results
    for text, true_label in data:
        # Obtain prediction from the classifier
        # The output is a list containing a dictionary; extract the 'label' value.
        prediction = classifier(text)
        predicted_label = prediction[0]['label'] # e.g., 'POSITIVE' or 'NEGATIVE'

        print(f"Text:        {text}")
        print(f"True Label:  {true_label}")
        print(f"Predicted:   {predicted_label}")
        print("-" * 20) # Separator
    ```

4.  **Output**: The output will first indicate model loading, followed by the analysis results for each sentence. Each result block shows the input text, the provided true label, and the label predicted by the DistilBERT model. Observe the correspondence and any discrepancies between true and predicted labels.

---

**Section 1.2: Interactive GUI with Gradio**

1.  **Description**:
    *   **Objective**: Create an interactive web interface using Gradio. This interface will accept user-provided text input and display the model's sentiment prediction.

2.  **Steps**:

    *   **Step 1: Import libraries:** Import `gradio` (commonly aliased as `gr`) and the `pipeline` from `transformers`.
    *   **Step 2: Load the model:** Ensure the `distilbert-base-uncased` sentiment analysis pipeline is loaded (as done in Section 1.1).
    *   **Step 3: Define a prediction function:** Create a Python function that accepts a text string as input, uses the `classifier` to predict its sentiment, and returns the predicted label string (`'POSITIVE'` or `'NEGATIVE'`). Include handling for empty input.
    *   **Step 4: Create the Gradio interface:** Use `gr.Interface` to configure the web interface. Specify the prediction function (`fn`), the input component type (`inputs` - a Textbox), and the output component type (`outputs` - another Textbox). Optional parameters like `title`, `description`, and `live` can configure the appearance and behavior.
    *   **Step 5: Launch the interface:** Call the `.launch()` method on the created interface object. This starts the Gradio application within the Colab notebook environment.

3.  **Code**: Execute the following code in a new Colab cell:

    ```python
    # Step 1: Import Gradio and pipeline
    import gradio as gr
    from transformers import pipeline

    # Step 2: Load the sentiment analysis model
    # If the 'classifier' object from Section 1.1 is still available, this line can be omitted.
    # Otherwise, ensure the model is loaded.
    print("Loading model (if needed)...")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    print("Model ready.")

    # Step 3: Define the function for Gradio
    def classify_sentiment_interface(text_input):
        # Basic input validation
        if not text_input:
            return "Input text is required."
        # Get prediction from the loaded classifier
        prediction = classifier(text_input)
        # Return the predicted label
        return prediction[0]['label']

    # Step 4: Create the Gradio interface object
    iface = gr.Interface(
        fn=classify_sentiment_interface,    # Function to call
        inputs=gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text"), # Input configuration
        outputs=gr.Textbox(label="Predicted Sentiment"), # Output configuration
        title="Sentiment Analysis Interface", # Interface title
        description="Input text to get sentiment prediction (POSITIVE/NEGATIVE) using DistilBERT.", # Interface description
        live=True                           # Enable real-time prediction updates as user types
    )

    # Step 5: Launch the Gradio interface
    print("\nLaunching Gradio Interface...")
    iface.launch()
    ```

4.  **Interface Interaction**: After execution, a Gradio interface will appear in the cell's output area.
    *   Enter text into the "Input Text" field (e.g., "The documentation was clear and concise.").
    *   The "Predicted Sentiment" field will update to show the model's output (e.g., `POSITIVE`).
    *   Test with various positive, negative, and potentially neutral inputs to observe the model's behavior.

---

**End of Part 1**

*   **Summary**: In this part, you have used the Hugging Face `transformers` library to load and apply a pre-trained DistilBERT model for sentiment analysis on sample text. You also constructed an interactive web interface using Gradio to allow real-time sentiment prediction based on user input. This demonstrates a basic workflow for text classification.

*   **Further Investigation (Optional)**:
    *   The Hugging Face NLP Course provides comprehensive material on Transformers and related concepts. Refer to [Chapter 1, Section 3: "Transformers, what can they do?"](https://huggingface.co/learn/nlp-course/chapter1/3) for an overview of various tasks addressable by the `pipeline` function. You could modify the `pipeline` call in the code (e.g., `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")`) to experiment with other NLP tasks directly.

---
## Part 2: