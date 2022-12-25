import pandas
import seaborn
import sklearn.metrics as metrics
import matplotlib.pyplot as pyplot

# Reading the Dataset:
read_data = pandas.read_csv("spam_ham_dataset.csv")

# Collecting Information on given Data:
print(read_data.groupby("Label").count())

# Mapping Label to Numbers:
read_data["Label_Mapped"] = read_data["Label"].map(lambda x: 1 if x == "spam" else 0)

# Data Pre-Processing:

read_data_processed = read_data.copy()
# Removing all the Punctuation from the Text:
read_data_processed["Text"] = read_data_processed["Text"].str.replace("\W+", " ").str.replace("\s+", " ").str.strip()
# Converting all the words to Lower-Case:
read_data_processed["Text"] = read_data_processed["Text"].str.lower()
# Splitting all the words into Lists:
read_data_processed["Text"] = read_data_processed["Text"].str.split()

# Splitting Data into Training and Testing Parts:

training_data = read_data_processed.sample(frac=0.8, random_state=1).reset_index(drop=True)
testing_data = read_data_processed.drop(training_data.index).reset_index(drop=True)
training_data = training_data.reset_index(drop=True)

# Data After Splitting =>
print("\nTraining Data Statistics =>")
print("\nSize of Training Data => ", training_data.shape)
print(training_data["Label"].value_counts() / training_data.shape[0] * 100)

print("\nTesting Data Statistics =>")
print("\nSize of Testing Data => ", testing_data.shape)
print(testing_data["Label"].value_counts() / testing_data.shape[0] * 100)

# Preparing the Vocabulary:

vocabulary = list(set(training_data["Text"].sum()))
print("\nVocabulary length => ", len(vocabulary))

# Calculating the Frequencies of each word:
word_count = pandas.DataFrame([
    [row[1].count(word) for word in vocabulary] for _, row in training_data.iterrows()], columns=vocabulary)

training_data = pandas.concat([training_data.reset_index(), word_count], axis=1).iloc[:, 1:]

# Implementation of Naive-Bayes Theorem =>
# Calculating values for Bayes' Theorem:

alpha = 1
vocab_size = len(training_data.columns) - 2
spam_probability = training_data["Label"].value_counts()["spam"] / training_data.shape[0]
ham_probability = training_data["Label"].value_counts()["ham"] / training_data.shape[0]
num_words_spam = training_data.loc[training_data["Label"] == "spam", "Text"].apply(len).sum()
num_words_ham = training_data.loc[training_data["Label"] == "ham", "Text"].apply(len).sum()


# Defining functions to calculate Probability of an E-Mail to be SPAM or HAM:
# Probability of a word to be SPAM:
def prob_spam(word):
    if word in training_data.columns:
        return (training_data.loc[training_data["Label"] == "spam", word].sum() + alpha) / (
                    num_words_spam + alpha * vocab_size)
    else:
        return 1


# Probability of a word to be HAM:
def prob_ham(word):
    if word in training_data.columns:
        return (training_data.loc[training_data["Label"] == "ham", word].sum() + alpha) / (
                    num_words_ham + alpha * vocab_size)
    else:
        return 1


spam_prob_list = []


# Classifier Function to classify words into SPAM or HAM:
def spam_ham_classifier(input_message):
    input_message_spam_prob = spam_probability
    input_message_ham_prob = ham_probability

    for word in input_message:
        input_message_spam_prob *= prob_spam(word)
        input_message_ham_prob *= prob_ham(word)

    spam_prob_list.append(input_message_spam_prob)
    if input_message_spam_prob > input_message_ham_prob:
        return "spam"
    else:
        return "ham"


# Implementation of Classifier Function to predict SPAM E-Mails:
testing_data["PREDICTION"] = testing_data["Text"].apply(spam_ham_classifier)
print("\n\nThe Final Result of Prediction is => ")
print(testing_data.head(10))

# Accuracy Calculation:
correct_predictions = (testing_data["PREDICTION"] == testing_data["Label"]).sum() / testing_data.shape[0] * 100
print("\nAccuracy => ", correct_predictions)

# Confusion Matrix:
confusion_matrix = metrics.confusion_matrix(testing_data["Label"], testing_data["PREDICTION"])
seaborn.heatmap(confusion_matrix, cmap="Blues", annot=True, cbar_kws={"label": "Color Bar"}, xticklabels=[0, 1],
                yticklabels=[0, 1])
pyplot.title("Confusion Matrix:")
pyplot.xlabel("PREDICTIONS")
pyplot.ylabel("ACTUAL")
pyplot.show()

# Sensitivity, Specificity and Accuracy Graph:
# Initially, Cut-Off for SPAM is spam_prob > 0.5

# Mapping Predicted values to Numbers: HAM == 0, SPAM == 1
testing_data["PREDICTION_NUM"] = testing_data["PREDICTION"]
testing_data["PREDICTION_NUM"] = testing_data["PREDICTION_NUM"].map({"ham": 0, "spam": 1})

# Creating Columns with different Probability Cut-Offs:
# Adding the Spam Probability Column in testing_data:
testing_data["Spam Probability"] = spam_prob_list

numbers = [float(x) / 10 for x in range(10)]
for i in numbers:
    testing_data[i] = testing_data["Spam Probability"].map(lambda x: 1 if x > i else 0)

# Creating a DataFrame for Cut-Off values:
cutoff_DF = pandas.DataFrame(columns=["Probability", "Accuracy", "Sensitivity", "Specificity"])

num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in num:
    confusion_matrix_temp = metrics.confusion_matrix(testing_data["Label_Mapped"], testing_data[i])
    total = sum(sum(confusion_matrix_temp))
    TP = confusion_matrix_temp[1][1]
    FP = confusion_matrix_temp[0][1]
    FN = confusion_matrix_temp[1][0]
    TN = confusion_matrix_temp[0][0]

    Accuracy = (TP + TN) / total
    Specificity = TN / (TN + FP)
    Sensitivity = TP / (TP + FN)
    cutoff_DF.loc[i] = [i, Accuracy, Sensitivity, Specificity]

# Plotting the Graph:
cutoff_DF.plot.line(x="Probability", y=["Accuracy", "Sensitivity", "Specificity"])
pyplot.title("Accuracy, Sensitivity and Specificity:")
pyplot.show()
