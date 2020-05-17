import numpy as np
import pandas as pd
import random
import re
import json
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter
from datetime import date
from fpdf import FPDF

import sys
import warnings
import time
import os
import ast
import glob
import operator


import locale
locale.setlocale(locale.LC_ALL, 'en_US')
font_style = 'Arial'


class CustomPDF(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font(font_style, 'I', 8)

        # Add a page number
        page = 'Page ' + str(self.page_no())
        self.cell(0, 10, page, 0, 0, 'C')


warnings.filterwarnings("ignore")
pdf = CustomPDF()
pdf.set_font(font_style, size=10)
pdf.add_page()
image_index = 0


def convert_number_format(d):
    return locale.format("%d", d, grouping=True)


# Remove rows containing null values in all columns
def remove_rows_containg_all_null_values(df):
    idx = df.index[~df.isnull().all(1)]
    df = df.ix[idx]
    return df


# Read input donation file
def read_input_file(file_path):
    file_name = file_path.split('/')[-1]
    extension = file_name.split(".")[-1]
    if extension == "csv":
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    elif (extension == "xlsx") | (extension == "xls"):
        return pd.read_excel(file_path, encoding="ISO-8859-1")
    else:
        print("{} file format is not supported".format(extension))


# Identify columns for file stored in DataStore
def identify_years_columns(file_name):
    mapping_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "column_name_mapping.json"))
    with open(mapping_file_path) as jsonfile:
        data = json.load(jsonfile)
    for k, v in data.items():
        if k in file_name:
            return v
    return "[]"


# Identify text columns
def identify_info_columns(df, donation_columns):
    column_names = df.columns
    return [col for col in column_names if col not in donation_columns]


# Remove column contains 80% unique values and more than 50% null values
def remove_columns_unique_values(df, column_names):
    final_col_ = []
    number_of_sample = df.shape[0]
    for col in column_names:
        # print("Column: {}, Null count: {}".format(col, df[col].isnull().sum()))
        if df[col].isnull().sum() <= number_of_sample/2:
            final_col_.append(col)
    final_col = []
    for col in final_col_:
        # print("Column: {}, Unique count: {}".format(col, df[col].unique().shape[0]))
        if (df[col].unique().shape[0] <= number_of_sample*0.8) and (df[col].unique().shape[0] != 1):
            final_col.append(col)
    final_df = df[final_col]
    final_df.fillna("", inplace=True)
    return final_df


# Identify unique categorical columns
def identify_categorical_columns(df, column_names):
    cat_col_ = []
    for col in column_names:
        if df[col].unique().shape[0] <= 5:
            cat_col_.append(col)
    return cat_col_


# Regex function to clean input text
def text_processing(text):
    pre_text=[]
    for x in text:
        x = re.sub('[^a-zA-Z.\d\s]', '', x)
        x = re.sub(' +', ' ', x)
        x = str(x.strip()).lower()
        pre_text.append(x)
    return pre_text


# Convert text to numeric values using Tf-IDF
def feature_extraction(df_info):
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    unique_features = len(Counter([" ".join(x for x in processed_text)][0].split()).keys())
    feature_count = int(0.5*unique_features)
    if feature_count <= 1000:
        feature_count = 1000
    elif feature_count >= 3000:
        feature_count = 3000
    else:
        feature_count = int(0.5*unique_features)
    print("unique features {} and feature count {}".format(unique_features, feature_count))
    vectorizer = TfidfVectorizer(max_features=feature_count)
    X = vectorizer.fit_transform(processed_text)
    tfidf_matrix = X.todense()
    feature_names = vectorizer.get_feature_names()
    return processed_text, tfidf_matrix, feature_names, df_info, vectorizer


# Clean donation columns by keeping only digits
def clean_donation(donation):
    donation = ''.join(c for c in donation if (c.isdigit()) | (c == "."))
    if donation == "":
        return "0"
    else:
        return donation


# Identify target value for each record
def process_donation_columns(df, donation_columns, no_donation_columns, skewed_target_value):
    if no_donation_columns:
        donation_columns = ast.literal_eval(donation_columns)
    elif skewed_target_value:
        donation_columns = ast.literal_eval(donation_columns)

    donation_columns = df[donation_columns].fillna("0")
    donation_columns = donation_columns.astype(str)

    for col in donation_columns.columns:
        donation_columns[col] = donation_columns[col].apply(lambda x: clean_donation(x))
        donation_columns[col] = donation_columns[col].astype(float)

    donation_columns = donation_columns.astype(int)

    no_of_col = donation_columns.shape[1]

    def identify_target(x):
        non_zero_col = 0
        for col in donation_columns.columns:
            if x[col] >= 1:
                non_zero_col += 1
        return non_zero_col

    donation_columns['donation_non_zero'] = donation_columns.apply(lambda x: identify_target(x), axis=1)
    col_threshold = int(no_of_col/2.)
    donation_columns['target'] = donation_columns['donation_non_zero'].apply(lambda x: 1 if x > col_threshold else 0)
    del donation_columns['donation_non_zero']
    return donation_columns


# Generate correlation plot for donation columns
def generate_correlation(donation_columns, no_donation_columns, skewed_target_value):
    if no_donation_columns:
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="# Correlation Plot")
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Correlation Plot: Please note that Correlation is calculated based on the similar "
                                       "donor datasets stored in Predict Me's server. These datasets are used to find "
                                       "common donor attributes to maximize the model performance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="NOTE: The uploaded donor file is missing donation information (amount) required"
                                       " for plotting a Correlation Matrix.")
        pdf.ln(2)
    elif skewed_target_value:
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="# Correlation Plot")
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Correlation Plot: The uploaded donor file has an imbalanced dataset. More than 98% "
                                       "of your sample belongs to one class (0 or 1 Target Value) that make up a large "
                                       "proportion of the data.")
        pdf.ln(0.5)

        pdf.multi_cell(h=5.0, w=0, txt="Please note that Correlation is calculated based on the similar donor datasets "
                                       "stored in Predict Me's server. These datasets are used to avoid imbalanced data"
                                       " issues and find common donor attributes to maximize the model performance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="NOTE: The uploaded donor file is missing text values (attributes) required for "
                                       "plotting a Correlation.")
        pdf.ln(2)
    else:
        pdf.ln(2)
        sn.set(font_scale=2)
        fig, ax = plt.subplots(figsize=(25, 25))
        ax = sn.heatmap(donation_columns.corr().round(2).replace(-0, 0), annot=True)
        plt.title('Correlation Plot', fontsize=45)
        global image_index
        plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
        plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
        pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=175, h=150)
        image_index += 1


# Calculate sum of feature weights
def get_feature_weights(feature_list, feature_dict):
    sum_ = 0
    for f in feature_list:
        sum_ += feature_dict.get(f.lower(), 0)
    return sum_


# Plot feature importance for each column
def calculate_feature_importance(df_info, feature_names, feature_value, no_donation_columns, skewed_target_value):
    feature_dict = {i: abs(j) for i, j in zip(feature_names, feature_value)}
    info_columns = list(df_info.columns)
    info_columns.remove('comb_text')
    
    feature_dict_={}
    for col in info_columns:
        feat = [x.split() for x in df_info[col]]
        flat_list = [item for sublist in feat for item in sublist]
        feature_dict_[col]=get_feature_weights(text_processing(flat_list), feature_dict)
        
    total_sum = sum(feature_dict_.values())
    
    feature_imp = list(feature_dict_.values())/(total_sum/100)
    feature_columns = list(feature_dict_.keys())
    sn.set(font_scale=2)
    sorted_idx = np.argsort(feature_imp)
    pos = np.arange(sorted_idx.shape[0]) + .5
    if no_donation_columns:
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="# Feature Importance Plot")
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Feature Importance Plot: Please note that Feature Importance is calculated based "
                                       "on the similar donor datasets stored in Predict Me's server. These datasets "
                                       "are used to find common donor attributes to maximize the model performance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="NOTE: The uploaded donor file is missing text values (attributes) required for "
                                       "plotting a Feature Importance.")
        pdf.ln(2)
    elif skewed_target_value:
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="# Feature Importance Plot")
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="Feature Importance Plot: The uploaded donor file has an imbalanced dataset. More "
                                       "than 98% of your samples belong to one class (0 or 1 Target Value) that make up"
                                       " a large proportion of the data.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="Please note that Feature Importance is calculated based on the similar donor "
                                       "datasets stored in Predict Me's server. These datasets are used to avoid "
                                       "imbalanced data issues and find common donor attributes to maximize the model "
                                       "performance.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="NOTE: The uploaded donor file is missing text values (attributes) required for "
                                       "plotting a Feature Importance.")
        pdf.ln(2)
    else:
        featfig = plt.figure(figsize=(10, 6))
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, sorted(feature_imp), align='center')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(feature_columns)[sorted_idx], fontsize=12)
        featax.set_xlabel('% Relative Feature Importance', fontsize=16)
        # featax.set_xticklabels(fontsize=12)
        plt.tight_layout()
        plt.title('Feature Importance Plot', fontsize=16)
        global image_index
        plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
        plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
        pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=170, h=102)
        image_index += 1
        pdf.ln(2)


# Generate classification report
def add_classification_report_table(y_test, y_pred):
    report=classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['f1-score'] = report_df['f1-score'].apply(lambda x: str(round(x, 2)))
    report_df['precision'] = report_df['precision'].apply(lambda x: str(round(x, 2)))
    report_df['recall'] = report_df['recall'].apply(lambda x: str(round(x, 2)))
    report_df['support'] = report_df['support'].apply(lambda x: str(convert_number_format(int(x))))
    report_df = report_df.rename(columns={"f1-score": "F1-score",
                                          "precision": "Precision",
                                          "recall": "Recall",
                                          "support": "Support"})
    report_df.reset_index(inplace=True)
    report_df['index'] = report_df['index'].replace({"0": "Non-donor class", "1": "Donor class",
                                                     "macro avg": "Macro avg", "weighted avg": "Weighted avg"})
    report_df = report_df.rename(columns={"index": "Index"})
    report_df = pd.DataFrame(np.vstack([report_df.columns, report_df]))
    report_df = report_df.values.tolist()
    del report_df[3]
    spacing=1.25
    col_width = pdf.w / 6
    row_height = pdf.font_size
    for row in report_df:
        for item in row:
            pdf.cell(col_width, row_height*spacing, txt=item, border=1, align="C")
        pdf.ln(row_height*spacing)


# Plot confusion matrix
def print_confusion_matrix_classification_report(y_test, y_pred, no_donation_columns, skewed_target_value):
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2), range(2))
    plt.figure(figsize=(15, 10))
    sn.set(font_scale=2.5) # for label size
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 30}) # font size
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tick_params(axis="both", which="both", labelsize="large")
    global image_index
    pdf.set_font(font_style, size=10)
    # pdf.multi_cell(h=5.0, w=0, txt="# Confusion Matrix Plot")
    if no_donation_columns:
        pdf.multi_cell(h=5.0, w=0, txt="Confusion Matrix Plot: Please note that the output displayed is based on the similar"
                                       " reference donor datasets stored in Predict Me's server. These datasets are "
                                       "used to find common donor attributes to maximize the model performance.")
    if skewed_target_value:
        pdf.multi_cell(h=5.0, w=0, txt="Confusion Matrix Plot: The uploaded donor file has an imbalanced dataset. More "
                                       "than 98% of your samples belong to one class (0 or 1 Target Value) that make up"
                                       " a large proportion of the data.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="Please note that the output displayed is based on the similar reference donor "
                                       "datasets stored in Predict Me's server. These datasets are used to avoid "
                                       "imbalanced data issues and find common donor attributes to maximize the model "
                                       "performance.")

    pdf.ln(1)
    pdf.set_font(font_style, size=10)
    plt.title('Confusion Matrix Plot', fontsize=36)
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
    pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=112, h=75)
    image_index += 1
    pdf.ln(4)
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="# Classification Report Table")
    pdf.set_font(font_style, size=10)
    pdf.ln(4)
    add_classification_report_table(y_test, y_pred)
    pdf.ln(4)


# Calculates false postitive and true positive rate
def calculate_fpr_tpr(model, y_test, y_pred, X_test):
    try:
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
    return fpr, tpr, auc


# Plot ROC curve
def plot_roc_curve(roc_fpr, roc_tpr, roc_auc, top_5_models, no_donation_columns, skewed_target_value):
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="# Receiver Operating Characteristic (ROC) Curve")
    pdf.set_font(font_style, size=10)
    pdf.ln(2)
    pdf.multi_cell(h=5.0, w=0, txt="It is a plot of the false positive rate (x-axis) versus the true positive rate "
                                   "(y-axis). True positive rate or sensitivity ")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="describes how good the model is at predicting the positive class when the actual "
                                   "outcome is positive. False positive rate explains how often a positive class is "
                                   "predicted when the actual result is negative.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="A model with high accuracy is represented by a line that travels from "
                                   "the bottom left of the plot to the top left and then across the top to the top "
                                   "right and has Area Under Curve (AUC) as 1. A model with less accuracy is represented by "
                                   "a diagonal line from the bottom left of the plot to the top right and has an AUC "
                                   "of 0.5.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="We can compare multiple models using AUC value; the best model will have AUC "
                                   "close to 1.")
    pdf.ln(1)
    if no_donation_columns:
        pdf.multi_cell(h=5.0, w=0, txt="ROC: Please note that the output displayed is based on the similar reference "
                                       "donor datasets stored in Predict Me's server. These datasets are used to find "
                                       "common donor attributes to maximize the model performance.")
        pdf.ln(1)
    if skewed_target_value:
        pdf.multi_cell(h=5.0, w=0, txt="ROC: The uploaded donor file has an imbalanced dataset. More than 98% of your "
                                       "samples belong to one class (0 or 1 Target Value) that make up a large "
                                       "proportion of the data.")
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="Please note that the output displayed is based on the similar reference donor "
                                       "datasets stored in Predict Me's server. These datasets are used to avoid "
                                       "imbalanced data issues and find common donor attributes to maximize the model "
                                       "performance.")
        pdf.ln(1)
    pdf.ln(2)
    plt.figure(figsize=(15, 10))
    sn.set(font_scale=2)
    for model_name in top_5_models:
        fpr = roc_fpr.get(model_name)
        tpr = roc_tpr.get(model_name)
        auc = roc_auc.get(model_name)
        plt.plot(fpr, tpr, label="{} ROC (area = {})".format(model_name, round(auc, 2)))

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    global image_index
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
    pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=175, h=105)
    image_index += 1
    pdf.ln(6)


# Train 10 classifiers
def model_selection(X, y, X_pred, donation_columns, cat_col, no_donation_columns, skewed_target_value):
    models = [{'label': 'LogisticRegression', 'model': LogisticRegression()},
                {'label': 'RidgeClassifier', 'model': RidgeClassifier()},  # No predict_proba
                {'label': 'MultinomialNB', 'model': MultinomialNB()},
                {'label': 'ComplementNB', 'model': ComplementNB()},
                {'label': 'BernoulliNB', 'model': BernoulliNB()},
                {'label': 'DecisionTreeClassifier', 'model': DecisionTreeClassifier()},
                {'label': 'SGDClassifier', 'model': SGDClassifier(loss='log')},
                {'label': 'PassiveAggressiveClassifier', 'model': PassiveAggressiveClassifier()},  # No predict_proba
                {'label': 'LinearSVC', 'model': LinearSVC()},  # No predict_proba
                {'label': 'RandomForestClassifier', 'model': RandomForestClassifier()}]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    if (not no_donation_columns) & (not skewed_target_value):
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="              a. 80% ({}) of data used for training the model".format(
            convert_number_format(X_train.shape[0])))
        pdf.ln(0.5)
        pdf.multi_cell(h=5.0, w=0, txt="              b. 20% ({}) of data used for testing the model".format(
            convert_number_format(X_test.shape[0])))
        pdf.ln(0.5)

    test_list = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    if no_donation_columns:
        pdf.multi_cell(h=5.0, w=0, txt="     2. Donation Columns: The uploaded donor file is missing donation "
                                       "information (amount) to show donation column(s).")
        pdf.ln(0.5)
    elif skewed_target_value:
        pdf.multi_cell(h=5.0, w=0, txt="     2. Donation Columns: The uploaded donor file has an imbalanced dataset to"
                                       " show donation column(s). More than")
        pdf.ln(0.25)
        pdf.multi_cell(h=5.0, w=0, txt= "         98% of your sample belongs to one class (0 or 1 Target Value) that make up a "
                                        "large proportion of the data.")
        pdf.ln(0.5)
    else:
        pdf.multi_cell(h=5.0, w=0, txt="     2. Donation Columns:")
        pdf.ln(0.5)
        for i in range(len(donation_columns)):
            pdf.multi_cell(h=5.0, w=0, txt="              {}. {}".format(test_list[i], donation_columns[i]))
            pdf.ln(0.3)
        pdf.ln(0.5)

    if len(cat_col) > len(test_list):
        cat_col = random.sample(cat_col, len(test_list))
    if len(cat_col) != 0:
        pdf.multi_cell(h=5.0, w=0, txt="     3. Categorical Columns:")
        pdf.ln(0.5)
        for i in range(len(cat_col)):
            pdf.multi_cell(h=5.0, w=0, txt="              {}. {}".format(test_list[i], cat_col[i]))
            pdf.ln(0.3)
    else:
        pdf.multi_cell(h=5.0, w=0, txt="     3. Categorical Columns: No categorical columns identified on the uploaded "
                                       "donor file.")
        pdf.ln(0.5)

    print_steps_taken()
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="C. Important Terms Used in Predictive Modeling")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    pdf.multi_cell(h=5.0, w=0, txt="     1. F1-score: It is a harmonic mean of precision and recall.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     2. Precision: It is a fraction of correctly classified instances among all "
                                   "predicted instances.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     3. Recall: It is a fraction of correctly classified instances among all "
                                   "actual/valid instances.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     4. Support: Number of samples used for the experiment.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     5. Confusion Matrix Plot: It is a plot of the true count (x-axis) versus "
                                   "predicted count (y-axis) for both the classes")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         (donor and non-donor). The top left box represents the count of true "
                                   "negatives, the top right box represents the")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         count of false negatives, bottom left box represents the count of false "
                                   "positives and bottom right box represents")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         the count of true positives.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     6. Feature Importance Plot: Y-axis: feature present in input file and "
                                   "X-axis: relative % of feature importance.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     7. Correlation Plot: Correlation explains how one or more variables are "
                                   "related to each other.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     8. Probability Score: It is a probabilty (likelihood) of an individual to "
                                   "donate.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     9. Threshold Value: It is the threshold (cut-off) value used on a probability "
                                   "score to seperate a donor from a")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         non-donor.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     10. Predicted Classification (0 and 1): Classification value 1 indicates an "
                                   "individual likely to donate and classification")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         value 0 indicates an individual less likely to donate. They follow the "
                                   "threshold (cut-off) value logic.")
    pdf.ln(0.5)
    pdf.ln(3)

    plt.figure(figsize=(15, 10))
    model_f1_score={}
    classification_full_pred={}
    classification_full_pred_prob={}
    feature_importance_dict={}
    roc_fpr={}
    roc_tpr={}
    roc_auc={}
    y_test_dict={}
    y_pred_dict={}
    for ind, m in enumerate(models):
        start_time = time.time()
        model = m['model']
        if m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC', 'RidgeClassifier']:
            model = CalibratedClassifierCV(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        classification_full_pred[m['label']] = model.predict(X_pred)
        classification_full_pred_prob[m['label']] = model.predict_proba(X_pred)

        print("Classifier: {} and time(seconds): {}".format(m['label'], round(time.time()-start_time, 3)))
        print()
        model_f1_score[m['label']] = round(f1_score(y_test, y_pred, average='weighted'), 2)
        y_test_dict[m['label']] = y_test
        y_pred_dict[m['label']] = y_pred
        if m['label'] in ['DecisionTreeClassifier', 'RandomForestClassifier']:
            feature_value = model.feature_importances_[:-1]
        elif m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC', 'RidgeClassifier']:
            model = m['model']
            model.fit(X_train, y_train)
            feature_value = model.coef_[0][:-1]
        elif m['label'] in ['GaussianNB']:
            continue
        else:
            feature_value = model.coef_[0][:-1]

        feature_importance_dict[m['label']] = feature_value

        fpr, tpr, auc = calculate_fpr_tpr(model, y_test, y_pred, X_test)
        roc_fpr[m['label']] = fpr
        roc_tpr[m['label']] = tpr
        roc_auc[m['label']] = auc

    return model_f1_score, classification_full_pred, classification_full_pred_prob, feature_importance_dict, roc_fpr, \
           roc_tpr, roc_auc, y_test_dict, y_pred_dict


# Generate Prediction File with best classifier
def generate_prediction_file(df, model_f1_score, classification_full_pred, classification_full_pred_prob, y,
                             feature_importance_dict, roc_fpr, roc_tpr, roc_auc, y_test_dict, y_pred_dict,
                             feature_names, df_info, donation_columns_df, no_donations_columns, skewed_target_value):
    model_f1_score = {k: v for k, v in sorted(model_f1_score.items(), key=lambda item: item[1])}
    # Number of models we want in report, modify the count below
    top_5_model = sorted(model_f1_score, key=model_f1_score.get, reverse=True)[:1]
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="D. Best Fit Model Used in Predictive Modeling")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    pdf.multi_cell(h=5.0, w=0, txt="Best fit classifier (model) is selected (out of 10 classifiers) based on F1-score and used "
                                   "for prediction. Model identified the optimal threshold to separate classes (donor and "
                                   "non-donor). Following are F1-score, threshold and count of donor samples.")
    for ind, m in enumerate(top_5_model):
        prediction = classification_full_pred.get(m)
        prob = classification_full_pred_prob.get(m)
        probability_column_name = 'Model Name: {}: Donor Probability Score'.format(m)
        prediction_column_name = 'Model Name: {}: Donor Predicted Classification (>= Threshold Value)'.format(m)
        df[probability_column_name] = [round(prob[x][1], 2) for x in range(len(prob))]
        # df['non_donor_prob_{}'.format(m)] = [round(prob[x][0], 2) for x in range(len(prob))]
        if no_donations_columns | skewed_target_value:
            max_acc_threshold = [0.5]
        else:
            t_={}
            for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
                df[prediction_column_name] = df[probability_column_name].apply(lambda x: 1 if x >= t else 0)
                t_[t] = round(f1_score(list(df[prediction_column_name]), y), 3)
            print(m)
            t_sorted = sorted(t_.items(), key=operator.itemgetter(1))
            print(t_sorted)
            max_acc_threshold = t_sorted[-1]

        print("Threshold used: {}".format(max_acc_threshold[0]))
        df[prediction_column_name] = df[probability_column_name].apply(
            lambda x: 1 if x >= max_acc_threshold[0] else 0)
        donor_count = df[df[prediction_column_name] == 1].shape[0]
        donor_per = round((donor_count/df.shape[0])*100, 2)
        pdf.ln(2)
        pdf.set_font(font_style, 'BU', size=10)
        pdf.multi_cell(h=5.0, w=0, txt="Best fit model name: {}".format(m))
        pdf.set_font(font_style, size=10)
        pdf.ln(1)
        pdf.multi_cell(h=5.0, w=0, txt="        a. F1-score (accuracy score): {}".format(model_f1_score.get(m)))
        pdf.ln(0.75)
        pdf.multi_cell(h=5.0, w=0, txt="        b. Threshold used: {}".format(max_acc_threshold[0]))
        pdf.ln(0.75)
        pdf.multi_cell(h=5.0, w=0, txt="        c. Donor predicted: {}% ({} out of {})".format(
            donor_per, convert_number_format(donor_count), convert_number_format(df.shape[0])))
        pdf.ln(3)
        print_confusion_matrix_classification_report(y_test_dict.get(m), y_pred_dict.get(m), no_donations_columns,
                                                     skewed_target_value)

        calculate_feature_importance(df_info, feature_names, feature_importance_dict.get(m), no_donations_columns,
                                     skewed_target_value)

    plot_roc_curve(roc_fpr, roc_tpr, roc_auc, top_5_model, no_donations_columns, skewed_target_value)
    if donation_columns_df.shape[1] != 0:
        generate_correlation(donation_columns_df, no_donations_columns, skewed_target_value)
    return df, m


# Get tfidf featues for file found from DB (No donation columns present)
def get_tfidf_features(file_name):
    df = read_input_file(file_name)
    df = remove_rows_containg_all_null_values(df)
    df_info = remove_columns_unique_values(df, identify_info_columns(df, []))
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_text)
    return vectorizer.get_feature_names()


# Find similar files in DB (No donation columns present)
def find_similar_files(input_file):
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), input_file))
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "donation_amount_files"))
    input_features = get_tfidf_features(input_file)
    common_features = {}
    for file_name in glob.glob(directory_path+'/*.*'):
        if file_name != input_file:
            file_features = get_tfidf_features(file_name)
            print("Total features: {} for: {}".format(len(set(file_features)), file_name.split('/')[-1]))

            common_features_per = float(len(set(file_features) & set(input_features)))*100/len(set(file_features))
            common_features[file_name] = common_features_per 
            print("% of common features: {} for: {}".format(common_features_per, file_name))

    file_dict = {k: v for k, v in sorted(common_features.items(), key=lambda item: item[1])}
    x=sorted(file_dict, key=file_dict.get, reverse=True)
    return sorted(file_dict, key=file_dict.get, reverse=True)[0]


# Transform text values to tf-idf features
def transform_features(vectorizer, df_info):
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    X = vectorizer.transform(processed_text)
    tfidf_matrix = X.todense()
    return tfidf_matrix


# Print steps taken to run classifier in PDF
def print_steps_taken():
    pdf.ln(3)
    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="B. Running the Predictive Model: A Step by Step Guide")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    pdf.multi_cell(h=5.0, w=0, txt="     1. Read the input data file provided.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     2. Cleaning up of data: remove null rows and columns and impute missing values.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     3. Identifying columns containing categorical and textual data and converting "
                                   "it to numerical values. If a column has")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         less than or equal to five unique values, then it is identified as "
                                   "categorical value.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     4. Assigning Target Value: Target values are the dependent (predicted) "
                                   "variable. Total donation columns are")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         calculated to assign target value. For example, if 50% of the total "
                                   "donation columns have a donation amount")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         (> 0.00 value), the model assigns that row (record) as 1 otherwise 0.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     5. Splitting the dataset for training and testing to train a total of 10 "
                                   "different classifiers (for example Naive Bayes,")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         Logistic Regression and Random Forest).")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     6. Calculating Feature Importance for each classifier. Feature importance "
                                   "gives a score for each feature of your data.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     7. Plot Confusion Matrix and Classification report. A confusion matrix is a "
                                   "table that is used to describe the")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         performance of a model.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     8. Identifying and selecting the best fit classifier (model) using the "
                                   "F1-score. The F1-score is a measure of a test's")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         (model's) accuracy.")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     9. Receiver Operating Characteristic (ROC) Curve. ROC is a probability curve. "
                                   "It tells how much a model is capable")
    pdf.ln(0.25)
    pdf.multi_cell(h=5.0, w=0, txt="         of distinguishing between classes (donor and non-donor).")
    pdf.ln(0.5)
    pdf.multi_cell(h=5.0, w=0, txt="     10. Identifying the optimal threshold (accuracy of the model) and predict.")
    pdf.ln(3)


# Delete old plots from directory
def delete_old_plots():
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    files = glob.glob('{}/*.png'.format(plots_path))
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "logo"))
    pdf.image("{}/logo-new.png".format(logo_path), w=105, h=35, x=55)
    pdf.ln(2)
    today = date.today()
    today_date = today.strftime("%B %d, %Y")
    pdf.set_font(font_style, 'B', size=10)
    pdf.multi_cell(h=5.0, w=0, txt="Predictive Modeling Results", align="C")
    pdf.ln(0.5)
    pdf.set_font(font_style, 'B', size=8)
    pdf.multi_cell(h=5.0, w=0, txt="Report Date: {}".format(today_date), align="C")
    pdf.ln(3)
    delete_old_plots()
    file_path = sys.argv[1]
    donation_columns = ast.literal_eval(sys.argv[2])
    donor_df = read_input_file(file_path)

    pdf.set_font(font_style, 'BU', size=10)
    pdf.multi_cell(h=7.5, w=0, txt="A. Data Input Summary")
    pdf.set_font(font_style, size=10)
    pdf.ln(1)
    donor_df = remove_rows_containg_all_null_values(donor_df)
    no_donations_columns = False
    skewed_target_value = False
    donation_columns_df = process_donation_columns(donor_df, donation_columns, no_donations_columns,
                                                   skewed_target_value)
    postive_class = donation_columns_df[donation_columns_df['target'] == 1].shape[0]
    negative_class = donation_columns_df[donation_columns_df['target'] == 0].shape[0]

    pdf.multi_cell(h=5.0, w=0, txt="     1. Total Data Sample: {}".format(convert_number_format(donor_df.shape[0])))

    if (len(donation_columns) == 0) | ((postive_class <= (donation_columns_df.shape[0])*0.02) |
                                       (negative_class <= (donation_columns_df.shape[0])*0.02)):
        print("No donation columns present or data is skewed")
        print("donation columns {}".format(len(donation_columns)))
        print("positive class {} and negative class {}".format(postive_class, negative_class))
        if len(donation_columns) == 0:
            no_donations_columns = True
        elif ((postive_class <= (donation_columns_df.shape[0])*0.02) | (negative_class <=
                                                                       (donation_columns_df.shape[0])*0.02)):
            skewed_target_value = True
        new_file = find_similar_files(file_path)
        print(new_file)
        df = read_input_file(new_file)
        df = remove_rows_containg_all_null_values(df)
        donation_columns = identify_years_columns(new_file)
        donation_columns_df = process_donation_columns(df, donation_columns, no_donations_columns, skewed_target_value)
    else:
        df = donor_df

    info_columns = identify_info_columns(df, donation_columns)
    if len(info_columns) < 3:
        raise ValueError("In order for the model to run, please supply a minimum of three text columns on your donor "
                         "file.")
    df_info = remove_columns_unique_values(df, info_columns)
    if no_donations_columns | skewed_target_value:
        info_columns = identify_info_columns(donor_df, [])
        cat_col = identify_categorical_columns(donor_df, info_columns)
    else:
        cat_col = identify_categorical_columns(df, info_columns)

    processed_text, tfidf_matrix, feature_names, df_info, vectorizer = feature_extraction(df_info)
    y = list(donation_columns_df['target'])

    if no_donations_columns | skewed_target_value:
        X_pred = transform_features(vectorizer, donor_df)
        X_train = tfidf_matrix
    else:
        X_pred = tfidf_matrix
        X_train = tfidf_matrix
        del donation_columns_df['target']

    model_f1_score, classification_full_pred, classification_full_pred_prob, feature_importance_dict, roc_fpr, \
    roc_tpr, roc_auc, y_test_dict, y_pred_dict = model_selection(X_train, y, X_pred, donation_columns, cat_col,
                                                                 no_donations_columns, skewed_target_value)
    df_final, best_model = generate_prediction_file(donor_df, model_f1_score, classification_full_pred,
                                        classification_full_pred_prob, y, feature_importance_dict, roc_fpr, roc_tpr,
                                        roc_auc, y_test_dict, y_pred_dict, feature_names, df_info, donation_columns_df,
                                        no_donations_columns, skewed_target_value)

    prediction_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediction"))
    pdf_report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "PdfReport"))
    file_path = file_path.split("/")[-1]
    pdf.output("{}/{}_{}_{}_report.pdf".format(pdf_report_path, file_path.split(".")[0], best_model, today_date))
    df_final.to_csv("{}/{}_{}_{}_prediction.csv".format(prediction_path, file_path.split(".")[0], best_model,
                                                        today_date), index=None)

