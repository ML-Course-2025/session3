# Lab: Getting Started with Text Classification and Interactive NLP Interfaces

> [!IMPORTANT]
> **Document Your Learning Journey:** Throughout this lab, actively use **Text Cells** in your Colab notebook. When you encounter a new concept (like `pipeline`, `DistilBERT`, `Gradio`), a function, or a specific parameter:
> *   **Explain it in your own words:** What is it? What does it do?
> *   **State its purpose here:** Why are we using it in this specific step?
> *   **Reflect:** Note any observations, interesting results, or questions that arise.
> This documentation is crucial for solidifying your understanding. You are encouraged to use external resources (like documentation or your favorite LLM) to gather details, but **always synthesize and explain the concepts yourself and verify the information.**

---

## Part 1: Text Classification with Transformers and Gradio

### Introduction

**What are we doing?** In this first part, we'll dive into **text classification**, a fundamental task in Natural Language Processing (NLP). Specifically, we'll perform **sentiment analysis** – determining if a piece of text expresses a positive or negative emotion. 

**How are we doing it?** We'll leverage the power of the Hugging Face `transformers` library to easily load and use a pre-trained **DistilBERT** model. DistilBERT is a smaller, faster variant of the famous BERT model, making it efficient for many tasks. We'll then use the `gradio` library to build a simple, interactive web interface (GUI) for our sentiment analysis model, allowing anyone to try it out directly.

**Learning Objectives:**

*   Understand and apply the Hugging Face `pipeline` function for a common NLP task.
*   Load and utilize a specific pre-trained Transformer model (DistilBERT) for sentiment analysis.
*   Interpret the output of a sentiment analysis model.
*   Construct a basic web-based UI for an NLP model using Gradio.

**Tools:**

*   Google Colab (requires a Google Account)
*   Hugging Face `transformers` library
*   `gradio` library

---

### Setup: Preparing Your Colab Environment

First, we need to install the necessary Python libraries within our Google Colab environment.

**Action:** Execute the following command in a code cell.

```python
# Install the necessary libraries using pip, Python's package installer.
# transformers: Provides access to pre-trained models and the 'pipeline' utility.
# gradio: Used for creating interactive web UIs for ML models.
# The '-q' flag suppresses verbose installation output.
!pip install transformers gradio -q

print("Libraries installed successfully!")
```

**Explanation:** This command downloads and installs the specified libraries from the Python Package Index (PyPI) into your Colab session's virtual environment, making their functions available for use in subsequent code cells. The `transformers` library might automatically install other necessary packages it depends on.

> [!NOTE]
> Sometimes, after installing new libraries, you might need to restart the Colab runtime for the changes to take full effect. You can do this via the menu: `Runtime` -> `Restart runtime`.

---

### Section 1.1: Console-Based Sentiment Analysis

Let's start by using the model directly within our notebook to classify the sentiment of some predefined sentences.

#### 1.1.1 Importing the `pipeline`

**Action:** Import the core component we need from the `transformers` library.

```python
# Import the 'pipeline' function from the transformers library.
from transformers import pipeline

print("Pipeline function imported.")
```

**Explanation:** The `pipeline` is a high-level abstraction provided by Hugging Face. It simplifies the process of using pre-trained models for various tasks by handling the necessary preprocessing (like tokenization), model inference, and post-processing (like converting model outputs to human-readable labels) behind the scenes.

#### 1.1.2 Loading the Sentiment Analysis Pipeline

**Action:** Instantiate the pipeline for sentiment analysis, specifying the DistilBERT model.

```python
# Create an instance of the sentiment analysis pipeline.
# Argument 1: "sentiment-analysis" tells the pipeline which task we want to perform.
# Argument 2: optional e.g model="distilbert-base-uncased" specifies the exact pre-trained model to load.
#   - 'distilbert': A distilled (smaller, faster) version of BERT.
#   - 'base': Refers to the model size (smaller than 'large').
#   - 'uncased': Indicates the model was trained on lowercased text and will process input similarly.
print("Loading the sentiment analysis model (DistilBERT)... This might take a moment.")
#classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
classifier = pipeline("sentiment-analysis")
print("Model loaded and pipeline ready.")
```

**Explanation:** This code downloads the specified DistilBERT model (if not already cached) and sets up the pipeline object (`classifier`) configured for sentiment analysis. This object can now be called like a function to classify text.

#### 1.1.3 Preparing Sample Data

**Action:** Define a list of sample sentences along with their "ground truth" sentiment labels (what we consider the correct answer).

```python
# Define a list of tuples. Each tuple contains:
# (text_sample, true_sentiment_label)
data = [
    ("I love this product! It's amazing.", "positive"),
    ("This is the worst experience I've ever had.", "negative"),
    ("I feel great after using this service.", "positive"),
    ("The customer support was horrible and unhelpful.", "negative"),
    # This one is more neutral/mixed, often tricky for binary classifiers
    ("It's okay, not great but not terrible either.", "negative")
]

print(f"Sample data prepared with {len(data)} entries.")
```

**Explanation:** Having sample data with known labels allows us to test the model and see how well its predictions align with our expectations. Notice the last example is intentionally less clear-cut.

#### 1.1.4 Classifying Sentiments and Displaying Results

**Action:** Loop through the sample data, use the `classifier` pipeline to get predictions, and print a comparison.

```python
print("\n--- Analyzing Sentiments ---")

# Iterate through each text sample and its true label in our data list.
for text, true_label in data:
    # Call the classifier pipeline with the text.
    # It returns a list containing a dictionary (or dictionaries for more complex tasks).
    # For sentiment analysis, it's typically [{'label': 'POSITIVE/NEGATIVE', 'score': 0.XXX}]
    prediction_result = classifier(text)

    # Extract the predicted label (e.g., 'POSITIVE' or 'NEGATIVE') from the result.
    # We access the first element [0] because the pipeline returns a list.
    predicted_label = prediction_result[0]['label']
    # Extract the confidence score associated with the prediction.
    predicted_score = prediction_result[0]['score']

    # Print the results in a readable format.
    print(f"Text:        {text}")
    print(f"True Label:  {true_label}")
    print(f"Predicted:   {predicted_label} (Confidence: {predicted_score:.4f})") # Format score to 4 decimal places

    # Add a simple check for correctness
    if predicted_label.lower() == true_label.lower():
        print("Outcome:     Correct")
    else:
        print("Outcome:     Incorrect")
    print("-" * 30) # Print a separator line
```

**Explanation:** This loop demonstrates the core usage of the pipeline: pass text to the `classifier` object. We then extract the `label` and `score` from the returned dictionary. The `score` represents the model's confidence in its prediction. We compare the `predicted_label` (converted to lowercase for consistency) with the `true_label` to see if the model was correct for our samples.

**Observe the Output:** Examine the results carefully.
*   How accurate was the model on these examples?
*   Look at the confidence scores. Are they higher for clearly positive/negative sentences?
*   How did the model classify the "It's okay..." sentence? Why might this be challenging for a model trained primarily on strongly positive or negative examples?

> [!NOTE]
> **Think about it:** Add a **Text Cell** here to discuss your observations about the model's performance, especially regarding the neutral/ambiguous sentence. Why do you think it behaved that way?

---

### Section 1.2: Interactive GUI with Gradio

Now, let's make our sentiment classifier interactive using Gradio.

#### 1.2.1 Importing Gradio and Ensuring Pipeline Availability

**Action:** Import the `gradio` library and make sure our `classifier` pipeline from Section 1.1 is accessible.

```python
# Import the Gradio library, commonly aliased as 'gr'.
import gradio as gr

# We also need the 'pipeline' function if this section is run independently
# or after a runtime restart.
from transformers import pipeline

# Check if the 'classifier' variable exists from the previous section.
# If not, reload the pipeline. This prevents reloading if not necessary.
try:
    # A quick test to see if the variable exists and is usable.
    _ = classifier("Test sentence")
    print("Classifier pipeline already loaded.")
except NameError:
    print("Classifier pipeline not found, reloading...")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    print("Pipeline reloaded.")

```

**Explanation:** We import `gradio` to build the UI. The `try-except` block is a safeguard to ensure the `classifier` object (our loaded model pipeline) is available. If the variable `classifier` doesn't exist in the current session's memory (e.g., if you only run this cell or restarted the runtime), it catches the `NameError` and reloads the pipeline.

#### 1.2.2 Defining the Prediction Function for Gradio

**Action:** Create a Python function that takes text input and returns the sentiment prediction string. Gradio will use this function.

```python
# Define a function that Gradio will call.
# It must accept arguments corresponding to the Gradio inputs
# and return values corresponding to the Gradio outputs.
def predict_sentiment(text_input):
    """
    Takes a string input, performs sentiment analysis using the global 'classifier',
    and returns the predicted label ('POSITIVE' or 'NEGATIVE').
    Includes basic input validation.
    """
    # Check if the input is empty or just whitespace.
    if not text_input or text_input.strip() == "":
        return "Please enter some text."

    # Use the pre-loaded classifier pipeline.
    prediction = classifier(text_input)

    # Extract the label from the result.
    predicted_label = prediction[0]['label']
    predicted_score = prediction[0]['score'] # Get score too

    # Return a formatted string including the label and score
    return f"{predicted_label} (Confidence: {predicted_score:.4f})"

# Test the function (optional, but good practice)
print(predict_sentiment("Gradio makes interfaces easy!"))
print(predict_sentiment("")) # Test empty input
```

**Explanation:** This function, `predict_sentiment`, acts as the bridge between the Gradio interface and our model. Gradio will pass the text entered by the user into the `text_input` argument. The function then uses the `classifier` to get the prediction and returns the result as a string, which Gradio will display. We added basic validation for empty input and formatted the output to include the confidence score.

#### 1.2.3 Creating and Configuring the Gradio Interface

**Action:** Use `gr.Interface` to define the components and behavior of our web UI.

```python
# Create the Gradio interface object.
sentiment_interface = gr.Interface(
    fn=predict_sentiment, # The function to call when the user interacts.
    inputs=gr.Textbox(    # Defines the input component.
        lines=2,          # Sets the initial height of the textbox.
        placeholder="Enter your text here...", # Placeholder text inside the box.
        label="Input Text" # A label displayed above the input box.
    ),
    outputs=gr.Textbox(   # Defines the output component.
        label="Predicted Sentiment" # A label displayed above the output box.
    ),
    title="Sentiment Analysis with DistilBERT", # The title displayed at the top of the UI.
    description="Enter some English text and see if the DistilBERT model thinks it's positive or negative. Powered by Hugging Face and Gradio.", # A description below the title.
    # live=True # Setting live=True updates prediction as you type, but can be resource-intensive.
                # Without it, the user clicks 'Submit'. Let's keep it off for simplicity.
    #allow_flagging="never" # Disables Gradio's built-in feedback mechanism for this demo.
)

print("Gradio interface configured.")
```

**Explanation:** `gr.Interface` is the main class for creating Gradio apps.
*   `fn`: Specifies the Python function to execute (`predict_sentiment` in our case).
*   `inputs`: Defines the input UI component(s). Here, `gr.Textbox` creates a multi-line text input field with a label and placeholder text.
*   `outputs`: Defines the output UI component(s). Here, another `gr.Textbox` will display the result returned by our function.
*   `title`, `description`: Add context and instructions to the UI.
*   `allow_flagging`: Controls whether users can flag potentially incorrect predictions (useful for data collection, but disabled here).

#### 1.2.4 Launching the Gradio Interface

**Action:** Launch the interactive interface.

```python
# Launch the interface. This will create and display the interactive UI
# directly within the Colab output cell.
print("Launching Gradio interface...")
sentiment_interface.launch()
```

**Explanation:** Calling the `.launch()` method on the `gr.Interface` object starts a local web server (managed by Gradio) and displays the interactive UI in the output area below the code cell.

**Interact with the UI:**
*   Type or paste some text into the "Input Text" box.
*   Click the "Submit" button.
*   Observe the prediction displayed in the "Predicted Sentiment" box.
*   Try various inputs: clearly positive, clearly negative, neutral, questions, etc. How does the model react?

---

### Summary of Part 1

In this part, you successfully:
1.  Installed the necessary `transformers` and `gradio` libraries.
2.  Used the Hugging Face `pipeline` to load a pre-trained `distilbert-base-uncased` model for sentiment analysis.
3.  Tested the model's predictions on sample sentences directly in the notebook.
4.  Built and launched an interactive web interface using Gradio, allowing real-time sentiment prediction from user input.

This demonstrates a basic but powerful workflow for applying pre-trained NLP models and making them accessible.

---

## Part 2: Further Investigation: Exploring Other Pipelines

The Hugging Face `pipeline` function supports many NLP tasks beyond sentiment analysis. This part encourages you to explore these capabilities independently.

**Your Task:**

1.  **Explore Pipeline Options:** Visit the Hugging Face documentation on Pipelines ([https://huggingface.co/docs/transformers/main_classes/pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)) or review relevant course materials (like [HF Course Ch1, Sec 3](https://huggingface.co/learn/nlp-course/chapter1/3)). Identify at least **two different** pipeline tasks that interest you (e.g., `"text-generation"`, `"translation_en_to_fr"`, `"zero-shot-classification"`, `"question-answering"`, `"fill-mask"`, `"summarization"`). *Do not choose `"sentiment-analysis"` again.*

2.  **Experiment in Code Cells:**
    *   For **each** of your two chosen tasks:
        *   **Create a new Code Cell.**
        *   Import `pipeline`.
        *   Instantiate the pipeline for the chosen task. If you like, you can specify an appropriate model (check the documentation or Hub for suggestions, e.g., `model="gpt2"` for text generation, `model="Helsinki-NLP/opus-mt-en-fr"` for English-to-French translation). If you don't specify a model, `pipeline` often loads a default one.
        *   Prepare suitable input data for the task (e.g., a prompt for text generation, text to translate, a question and context for QA).
        *   Call the pipeline with your input and `print()` the result(s).
        *   **Add a Text Cell below your code:** Explain the task, the model you used (or the default), the input you provided, and interpret the output clearly. What did the model do? Was the result what you expected?

3.  **Build a Gradio Interface for One Task:**
    *   Select **one** of the two tasks you just experimented with.
    *   **Create a new Code Cell.**
    *   Adapt the Gradio code structure from **Section 1.2** to build an interactive interface for *this new task*. This will involve:
        *   Defining a new prediction function (similar to `predict_sentiment`) that calls your chosen pipeline.
        *   Configuring `gr.Interface` with appropriate `inputs` and `outputs` for the task (e.g., two `Textbox` inputs for QA, one `Textbox` output for translation). Refer to the Gradio documentation ([https://www.gradio.app/docs/interface](https://www.gradio.app/docs/interface)) for different component types if needed.
        *   Updating the `title`, `description`, and potentially adding `examples`.
    *   Launch the interface using `.launch()`.
    *   **Add a Text Cell below your code:** Describe the interface you built, explain any significant changes you made to the Gradio code compared to Section 1.2, and comment on how well the interactive demo works for your chosen task.

**Deliverable:** Your final Colab notebook should contain all executed code cells (with their outputs visible) and your insightful Text Cells explaining concepts, documenting your experiments in Part 2, and reflecting on the results.

