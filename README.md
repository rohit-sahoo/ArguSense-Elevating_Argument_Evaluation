# ArguSense: Elevating Argument Evaluation using NLP

## Overview

Automated writing feedback tools often struggle to accurately identify key writing structures, such as thesis statements, claims, and supporting evidence, in essays. This project aims to enhance automated writing feedback by developing a model that accurately recognizes various writing components. By utilizing deep learning techniques and natural language processing (NLP), the project creates a model capable of analyzing text and identifying different writing components with varying levels of accuracy.

## Project Structure

![Input](https://drive.google.com/uc?export=view&id=1pzWSdBEYw3Sg4YhrdV_JxqV3zWVb6c9y)

![Output](https://drive.google.com/uc?export=view&id=1AeUoN5Tl5SlzZc7WpKfBPlRyyMN_JiUD)

### Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- Python 3.x
- TensorFlow 2.x
- Hugging Face Transformers library
- Pandas
- NumPy
- Matplotlib
- SpaCy
- Pytorch

### Dataset:
The dataset presented here contains argumentative essays. These essays were annotated by expert raters for discourse elements commonly found in argumentative writing:

1. Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
2. Position - an opinion or conclusion on the main question
3. Claim - a claim that supports the position
4. Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
5. Rebuttal - a claim that refutes a counterclaim
6. Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
7. Concluding Statement - a concluding statement that restates the claims

#### Dataset Link: 

1. [Dataset - Part 1: Identifying Sections in the Argument Essays](https://www.kaggle.com/competitions/feedback-prize-2021/data)

2. [Dataset - Part 2: Argument Essays Effectiveness](https://www.kaggle.com/competitions/feedback-prize-effectiveness/data) 


## Part 1: Identifying Sections in the Argument Essays:

Creating a model that accurately recognizes various writing components, including thesis statements, claims, and supporting evidence.

### Data Preparation and Preprocessing

The project uses a Python script to preprocess training data from a CSV file. The data contains essays or written texts with annotations for different writing components, such as Leads, Positions, Claims, Evidence, Concluding Statements, Counterclaims, and Rebuttals. The script tokenizes the text data using the Longformer tokenizer, creating token arrays for input and attention masks.

### Model Architecture

The model architecture consists of a deep learning model with an attention mechanism. It utilizes a pre-trained transformer model (Longformer) as a backbone and adds additional dense layers for classification. The model is designed to predict the presence of various writing components within the input text.

### Training and Evaluation

The model is trained using categorical cross-entropy loss and evaluated using categorical accuracy metrics. The validation dataset is used to assess the model's performance on recognizing different writing components. Evaluation metrics, including precision scores, are provided for each writing component category.

### Results and Conclusion

The project's results indicate the model's performance on different writing components. The precision scores vary across categories:

- Highest recognition rates are observed for Lead and Concluding Statement components.
- Moderate recognition rates are achieved for Position, Evidence, and Claim components.
- Lower recognition rates are observed for Counterclaim and Rebuttal components.
- The overall model performance, as measured by precision across all components, is around 63.4%.

## Usage

### Data Preparation:

1. Place the CSV file containing annotated essays in the appropriate directory.
2. Ensure that the CSV file is formatted with the necessary columns for text data and annotated components.

### Tokenization and Preprocessing:

1. Run the provided code snippet to preprocess training data and tokenize text.
2. Generate binary arrays indicating the presence of different writing components.

### Model Training:

1. Utilize the provided code to build and train the deep learning model using the Longformer architecture.
2. Monitor training progress and evaluate the model using validation data.

### Evaluation and Results:
**Evaluation Process**
For each sample in the dataset:

1. All ground truth and prediction instances associated with a particular class are compared.
2. The overlap between the ground truth and prediction is measured. An overlap is considered to occur when the shared portion of word indices is equal to or exceeds a threshold of 0.5.
3. Additionally, the overlap between the prediction and the ground truth is computed using the same threshold of 0.5.

Determining True Positives, False Positives, and False Negatives

1. If both the overlap between ground truth and prediction and the overlap between prediction and ground truth are greater than or equal to 0.5, the prediction is considered a match and marked as a true positive.
2. In cases where multiple matches exist, the match with the highest pair of overlaps is selected.
3. Unmatched ground truths are identified as false negatives, indicating instances where the model did not correctly identify a class.
4. Similarly, unmatched predictions are labeled as false positives, representing instances where the model incorrectly predicted a class.

**Accuracy:**
1. Lead 0.8063284233496999
2. Position 0.6841560234725578
3. Claim 0.6057328285559762
4. Evidence 0.6816788493279887
5. Concluding Statement 0.7827050997782705
6. Counterclaim 0.4854732895970009
7. Rebuttal 0.39030955585464333
   
Overall 0.6337691528480197

### Future Work:

- Consider refining the model architecture and hyperparameters for improved accuracy.
- Explore techniques to handle components with lower recognition rates, such as Counterclaims and Rebuttals.

## Part 2: Predicting the Effectiveness of the Arguments

In the context of this project, I aimed to develop a machine-learning model capable of classifying argumentative elements in student writing. The goal was to categorize these elements as "effective," "adequate," or "ineffective," which would provide valuable insights into the quality of argumentation in essays.

My primary task involved designing a classification system that could accurately determine the effectiveness of argumentative elements present in student-written essays. To enhance the model's performance, I decided to leverage a state-of-the-art language model, DeBERTa, which is known for its advanced contextual understanding and fine-tuning capabilities. The dataset consisted of labeled examples where each essay was associated with a specific label indicating its argumentation quality. To accomplish this task, I needed to implement several steps, including data preparation, text tokenization, dataset splitting, model training, and evaluation.

### Data Preparation: 

I started by importing essential libraries and checking the execution environment.

1. Import required libraries, including pathlib, os, zipfile
2. Read the training data into a Pandas DataFrame.
3. Convert label values to numerical values and rename the label column.

### Tokenization and Data Processing: 

1. Load a pre-trained tokenizer (AutoTokenizer) from Hugging Face's model repository.
2. Define a tokenization function to tokenize the input text data.
3. Tokenize the discourse text using the defined tokenizer and remove unnecessary columns from the dataset.


### Dataset Splitting: 
The dataset was divided into training and validation subsets to facilitate model evaluation. This step ensured that the model's performance could be assessed effectively.

### Training Setup: 

Parameters such as learning rate, batch size, weight decay, and number of epochs were defined. Evaluation metrics were selected, including log loss and softmax probabilities for score calculation.

1. Define learning rate (lr), batch size (bs), weight decay (wd), and number of epochs (epochs).
2. Import necessary metrics and functions for model evaluation.
3. Create a function to calculate the evaluation score using log loss and softmax probabilities.

### Model Training: 

A pre-trained DeBERTa model for sequence classification was loaded. This advanced model was fine-tuned on the training dataset using a Trainer object instantiated with specific training arguments and evaluation settings.

1. Define training arguments including output directory, learning rate, batch sizes, and others.
2. Load a pre-trained sequence classification model (AutoModelForSequenceClassification) with the specified number of labels.
3. Create a Trainer object to handle training and evaluation, passing the model, training arguments, datasets, and tokenizer.

### Testing and Evaluation:

For testing and evaluation, the project employs the multi-class logarithmic loss, commonly known as log loss. This metric is utilized to assess the quality of predictions made by the classification model.

In the context of this project:
1. Each row in the test dataset is assigned a true effectiveness label, representing the actual category of the argumentative element.
2. The model predicts the probabilities that an observation belongs to each effectiveness label category.
3. The log loss formula is employed to quantify the disparity between the predicted probabilities and the true labels.

Log loss:

log loss = -1/N * ∑(i=1 to N) ∑(j=1 to M) y<sub>ij</sub> * log(p<sub>ij</sub>)

Where:
- N is the number of rows in the test set.
- M is the number of class labels (in this case, the number of effectiveness categories).
- yij is 1 if observation i is in class j and 0 otherwise.
- pi,j is the predicted probability that observation i belongs to class j.

**Accuracy is 0.65 Log Loss for argument classification**



