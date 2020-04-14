import numpy as np
import pandas as pd
import re
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
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from matplotlib.backends.backend_pdf import PdfPages
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


class CustomPDF(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font('Arial', 'I', 8)

        # Add a page number
        page = 'Page ' + str(self.page_no())
        self.cell(0, 10, page, 0, 0, 'C')


warnings.filterwarnings("ignore")
pdf = CustomPDF()
pdf.set_font('Arial')
pdf.add_page()
image_index = 0


def remove_rows_containg_all_null_values(df):
    idx = df.index[~df.isnull().all(1)]
    df = df.ix[idx]
    return df


def read_input_file(file_path):
    file_name = file_path.split('/')[-1]
    extension = file_name.split(".")[-1]
    if extension == "csv":
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    elif extension == "xlsx":
        return pd.read_excel(file_path, encoding="ISO-8859-1")
    else:
        # print("{} file format is not supported".format(extension))
        pdf.multi_cell(h=5.0, w=0, txt="{} file format is not supported".format(extension))
        pdf.ln(5)


def identify_years_columns(df):
    column_names = df.columns
    return [col for col in column_names if "20" in col]


def identify_info_columns(df, donation_columns, identify_automatically=False):
    column_names = df.columns
    if identify_automatically:
        return [col for col in column_names if col not in donation_columns]
    else:
        return [col for col in column_names if "20" not in col]


## Remove column contains 80% unique values and more than 50% null values
def remove_columns_unique_values(df, column_names):
    final_col_ = []
    number_of_sample = df.shape[0]
    # print("Total training samples: {}".format(number_of_sample))
    for col in column_names:
        print("Column: {}, Null count: {}".format(col, df[col].isnull().sum()))
        # pdf.multi_cell(h=5.0, w=0, txt="Column: {}, Null count: {}".format(col, df[col].isnull().sum()))
        # pdf.ln(5)
        if df[col].isnull().sum() <= number_of_sample/2:
            final_col_.append(col)
    final_col = []
    # print("_____________\n")
    for col in final_col_:
        print("Column: {}, Unique count: {}".format(col, df[col].unique().shape[0]))
        # pdf.multi_cell(h=5.0, w=0, txt="Column: {}, Unique count: {}".format(col, df[col].unique().shape[0]))
        # pdf.ln(5)
        if (df[col].unique().shape[0] <= number_of_sample*0.8) and (df[col].unique().shape[0] != 1):
            final_col.append(col)
    final_df = df[final_col]
    final_df.fillna("", inplace=True)
    return final_df


def identify_categorical_columns(df, column_names):
    cat_col_ = []
    for col in column_names:
        if df[col].unique().shape[0] <= 5:
            cat_col_.append(col)
    return cat_col_


def text_processing(text):
    pre_text=[]
    for x in text:
        x = re.sub('[^a-zA-Z.\d\s]', '', x)
        x = re.sub(' +', ' ', x)
        x = str(x.strip()).lower()
        pre_text.append(x)
    return pre_text


def feature_extraction(df_info):
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    # print(processed_text[:5])
    unique_features = len(Counter([" ".join(x for x in processed_text)][0].split()).keys())
    feature_count = int(0.5*unique_features)
    if feature_count <= 1000:
        feature_count = 1000
    elif feature_count >= 10000:
        feature_count = 10000
    else:
        feature_count = int(0.5*unique_features)
    # pdf.write(5, "Number of features used: {}".format(feature_count))
    # pdf.multi_cell(h=5.0, w=0, txt="Number of features used: {}".format(feature_count))
    # pdf.ln(5)
    # print("Number of features used: {}".format(feature_count))
    vectorizer = TfidfVectorizer(max_features=feature_count)
    X = vectorizer.fit_transform(processed_text)
    tfidf_matrix = X.todense()
    feature_names = vectorizer.get_feature_names()
    return processed_text, tfidf_matrix, feature_names, df_info, vectorizer


def clean_donation(donation):
    donation = ''.join(c for c in donation if (c.isdigit()) | (c == "."))
    if donation == "":
        return "0"
    else:
        return donation


def process_donation_columns(df, donation_columns):
    if '2018 Donations Gave in the Area' in donation_columns:
        donation_columns.remove('2018 Donations Gave in the Area')
    if '2019 Donations Gave in the Area' in donation_columns:
        donation_columns.remove('2019 Donations Gave in the Area')
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
    # print(donation_columns['target'].value_counts())

    # positive_class_count = donation_columns[donation_columns['target'] == 1].shape[0]
    # negative_class_count = donation_columns[donation_columns['target'] == 0].shape[0]

    # pdf.multi_cell(h=5.0, w=0, txt="3. Positive sample count: {} and Negative sample count: {}".format(positive_class_count,
    #                                                                                                 negative_class_count))

    # pdf.multi_cell(h=5.0, w=0, txt=str(donation_columns['target'].value_counts()))
    # pdf.ln(5)
    return donation_columns


def generate_correlation(donation_columns):
    pdf.set_font('Arial', 'BU')
    pdf.multi_cell(h=5.0, w=0, txt="Correlation Plot")
    pdf.set_font('Arial')
    pdf.ln(3)
    pdf.multi_cell(h=5.0, w=0, txt="Correlation explains how one or more variables are related to each other.")
    pdf.ln(3)
    fig, ax = plt.subplots(figsize=(13, 13))
    ax = sn.heatmap(donation_columns.corr(), annot=True)
    global image_index
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
    pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=125, h=100)
    image_index += 1


def get_feature_weights(feature_list, feature_dict):
    sum_ = 0
    for f in feature_list:
        sum_ += feature_dict.get(f.lower(), 0)
    return sum_


def calculate_feature_importance(df_info, feature_names, feature_value):
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

    sorted_idx = np.argsort(feature_imp)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pdf.set_font('Arial', 'B')
    pdf.multi_cell(h=5.0, w=0, txt="# Feature Importance Plot")
    pdf.set_font('Arial')
    pdf.ln(5)
    featfig = plt.figure(figsize=(10, 6))
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, sorted(feature_imp), align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(feature_columns)[sorted_idx], fontsize=10)
    featax.set_xlabel('% Relative Feature Importance')
    plt.tight_layout()
    global image_index
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
    pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=160, h=96)
    image_index += 1
    pdf.ln(5)
    # plt.show()


def add_classification_report_table(y_test, y_pred):
    report=classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['f1-score'] = report_df['f1-score'].apply(lambda x: str(round(x, 2)))
    report_df['precision'] = report_df['precision'].apply(lambda x: str(round(x, 2)))
    report_df['recall'] = report_df['recall'].apply(lambda x: str(round(x, 2)))
    report_df['support'] = report_df['support'].apply(lambda x: str(x))
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


def print_confusion_matrix_classification_report(y_test, y_pred):
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2), range(2))
    print(df_cm)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 30}) # font size
    plt.xlabel("Predicted")
    plt.ylabel("True")
    global image_index
    pdf.set_font('Arial', 'B')
    pdf.multi_cell(h=5.0, w=0, txt="# Confusion Matrix Plot")
    pdf.set_font('Arial')
    pdf.ln(3)
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    plt.savefig("{}/temp_{}.png".format(plots_path, image_index))
    pdf.image("{}/temp_{}.png".format(plots_path, image_index), w=105, h=70)
    image_index += 1
    # plt.show()
    pdf.ln(5)
    pdf.set_font('Arial', 'B')
    pdf.multi_cell(h=5.0, w=0, txt="# Classification Report Table")
    pdf.set_font('Arial')
    pdf.ln(5)
    add_classification_report_table(y_test, y_pred)
    pdf.ln(5)
    # pdf.multi_cell(h=5.0, w=0, txt="___________________________\n")
    # pdf.ln(5)
    # print("classification report")
    # print(classification_report(y_test, y_pred))
    # print("___________________________\n")


def calculate_fpr_tpr(model, y_test, y_pred, X_test):
    try:
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
    # print("threshold", thresholds)
    return fpr, tpr, auc


def plot_roc_curve(roc_fpr, roc_tpr, roc_auc, top_5_models):
    pdf.set_font('Arial', 'BU')
    pdf.multi_cell(h=5.0, w=0, txt="Receiver Operating Characteristic (ROC) Curve")
    pdf.set_font('Arial')
    pdf.ln(5)
    pdf.multi_cell(h=5.0, w=0, txt="It is a plot of the false positive rate (x-axis) versus the true positive rate "
                                   "(y-axis). True positive rate or sensitivity describes how good the model is at "
                                   "predicting the positive class when the actual outcome is positive. False positive "
                                   "rate decribes how often a positive class is predicted when the actual outcome is "
                                   "negative. A model with high accuracy is represented by a line that travels from "
                                   "the bottom left of the plot to the top left and then across the top to the top "
                                   "right and has Area Under Curve (AUC) as 1. A model with less accuracy is represented by "
                                   "a diagonal line from the bottom left of the plot to the top right and has an AUC "
                                   "of 0.5. We can compare multiple models using AUC value, Best model will have AUC "
                                   "close to 1.")
    pdf.ln(5)
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
    pdf.ln(10)


def model_selection(X, y, X_pred):
    models = [{'label': 'LogisticRegression', 'model': LogisticRegression()},
                # {'label': 'GaussianNB', 'model': GaussianNB()}, #No feature importance
                {'label': 'MultinomialNB', 'model': MultinomialNB()},
                {'label': 'ComplementNB', 'model': ComplementNB()},
                {'label': 'BernoulliNB', 'model': BernoulliNB()},
                {'label': 'DecisionTreeClassifier', 'model': DecisionTreeClassifier()},
                {'label': 'SGDClassifier', 'model': SGDClassifier(loss='log')},
                {'label': 'PassiveAggressiveClassifier', 'model': PassiveAggressiveClassifier()}, #No predict_proba
                {'label': 'LinearSVC', 'model': LinearSVC()}, #No predict_proba
                {'label': 'RandomForestClassifier', 'model': RandomForestClassifier()}]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pdf.multi_cell(h=5.0, w=0, txt="4. 80 % of Data used for Training the model: {}".format(X_train.shape[0]))
    pdf.ln(3)
    pdf.multi_cell(h=5.0, w=0, txt="5. 20 % of Data used for Testing the model: {}".format(X_test.shape[0]))
    pdf.ln(3)
    print_steps_taken()
    pdf.set_font('Arial', 'BU')
    pdf.multi_cell(h=7.5, w=0, txt="C. Model Summary")
    pdf.set_font('Arial')
    pdf.ln(4)
    pdf.multi_cell(h=5.0, w=0, txt="Following terms are used while executing the models.")
    pdf.ln(3)
    pdf.multi_cell(h=5.0, w=0, txt="     1. F1-score: It is a harmonic mean of precision and recall.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="     2. Precision: It is a fraction of correctly classified instances among all "
                                   "predicted instances.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="     3. Recall: It is fraction of correctly classified instanes among all "
                                   "actual/true instances.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="     4. Support: Number of samples used for the experiment.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="     5. Confusion Matrix Plot: It is a plot of the true count (x-axis) versus "
                                   "predicted count (y-axis) for              both the classes. "
                                   "Top left box represents count of true negatives, top right "
                                   "box represents                  count of false negatives, bottom left box represents count of"
                                   " false positive and bottom right                 box represents count of true positives.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="     6. Feature Importance Plot: Y-axis: variable present in input file and "
                                   "X-axis: relative % of feature            importance.")
    pdf.ln(7.5)

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
        if m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC']:
            model = CalibratedClassifierCV(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        classification_full_pred[m['label']] = model.predict(X_pred) ## ADD New columns
        classification_full_pred_prob[m['label']] = model.predict_proba(X_pred)

        # print("Classifier: {} and time(seconds): {}".format(m['label'], round(time.time()-start_time, 3)))
        # print()
        # print("Classifier: {} and f1-score {}".format(m['label'], round(f1_score(y_test, y_pred, average='weighted'), 2)))
        model_f1_score[m['label']] = round(f1_score(y_test, y_pred, average='weighted'), 2)
        y_test_dict[m['label']] = y_test
        y_pred_dict[m['label']] = y_pred
        if m['label'] in ['DecisionTreeClassifier', 'RandomForestClassifier']:
            feature_value = model.feature_importances_[:-1]
        elif m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC']:
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

    # calculate_feature_importance(df_info, feature_value)
    # plot_roc_curve(roc_fpr, roc_tpr, roc_auc)
    # print_confusion_matrix_classification_report(y_test, y_pred)
    return model_f1_score, classification_full_pred, classification_full_pred_prob, feature_importance_dict, roc_fpr, \
           roc_tpr, roc_auc, y_test_dict, y_pred_dict


def generate_prediction_file(df, model_f1_score, classification_full_pred, classification_full_pred_prob, y,
                             feature_importance_dict, roc_fpr, roc_tpr, roc_auc, y_test_dict, y_pred_dict,
                             feature_names, df_info, donation_columns_df):
    model_f1_score = {k: v for k, v in sorted(model_f1_score.items(), key=lambda item: item[1])}
    top_5_model = sorted(model_f1_score, key=model_f1_score.get, reverse=True)[:5]
    # print(top_5_model, model_f1_score, classification_full_pred.keys(), classification_full_pred_prob.keys())
    pdf.set_font('Arial', 'BU')
    pdf.multi_cell(h=7.5, w=0, txt="D. Top 5 models used to predict")
    pdf.set_font('Arial')
    pdf.ln(4)
    pdf.multi_cell(h=5.0, w=0, txt="Top 5 classifiers are selected out of 10 classifiers based on F1-score and used for"
                                   " prediction. We identified optimal threshold to separate donor and non-donor "
                                   "classes. Following are f1-score, threshold and count of donor samples")
    for ind, m in enumerate(top_5_model):
        prediction = classification_full_pred.get(m)
        prob = classification_full_pred_prob.get(m)
        # df['2020_{}'.format(m)] = prediction
        df['donor_prob_{}'.format(m)] = [round(prob[x][1], 2) for x in range(len(prob))]
        df['non_donor_prob_{}'.format(m)] = [round(prob[x][0], 2) for x in range(len(prob))]
        t_={}
        for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, .7, .75, .8]:
            df['2020_{}'.format(m)] = df['non_donor_prob_{}'.format(m)].apply(lambda x: 0 if x > t else 1)
            t_[t] = round(f1_score(list(df['2020_{}'.format(m)]), y), 3)
        print(m)
        t_sorted = sorted(t_.items(), key=operator.itemgetter(1))
        print(t_sorted)
        max_acc_threshold = t_sorted[-1]
        print(max_acc_threshold[0], max_acc_threshold[1])
        print(df['2020_{}'.format(m)].value_counts())
        df['2020_{}'.format(m)] = df['non_donor_prob_{}'.format(m)].apply(
            lambda x: 0 if x > max_acc_threshold[0] else 1)
        donor_count = df[df['2020_{}'.format(m)] == 1].shape[0]
        donor_per = round((donor_count/df.shape[0])*100, 2)
        non_donor_count = df[df['2020_{}'.format(m)] == 0].shape[0]
        pdf.ln(3)
        pdf.set_font('Arial', 'BU')
        pdf.multi_cell(h=5.0, w=0, txt="Model {}. {}".format(ind+1, m))
        pdf.set_font('Arial')
        pdf.ln(3)
        pdf.multi_cell(h=5.0, w=0, txt="        a. F1-score: {}".format(model_f1_score.get(m)))
        pdf.ln(2.5)
        pdf.multi_cell(h=5.0, w=0, txt="        b. Threshold used: {}".format(max_acc_threshold[0]))
        pdf.ln(2.5)
        pdf.multi_cell(h=5.0, w=0, txt="        c. Donor predicted: {}% ({} out of {})".format(donor_per, donor_count,
                                                                                               df.shape[0]))
        pdf.ln(5)
        print_confusion_matrix_classification_report(y_test_dict.get(m), y_pred_dict.get(m))
        calculate_feature_importance(df_info, feature_names, feature_importance_dict.get(m))
        # pdf.multi_cell(h=5.0, w=0, txt="Non-Donor predicted: {}".format(non_donor_count))
        # pdf.ln(10)

    plot_roc_curve(roc_fpr, roc_tpr, roc_auc, top_5_model)
    if donation_columns_df.shape[1] != 0:
        generate_correlation(donation_columns_df)
    return df


def get_tfidf_features(file_name):
    df = read_input_file(file_name)
    df = remove_rows_containg_all_null_values(df)
    df_info = remove_columns_unique_values(df, identify_info_columns(df, [], False))
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_text)
    return vectorizer.get_feature_names()


def find_similar_files(input_file):
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), input_file))
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "donation_amount_files"))
    input_features = get_tfidf_features(input_file)
    common_features = {}
    for file_name in glob.glob(directory_path+'/*.csv'):
        if file_name != input_file:
            # print(file_name, input_file)
            pdf.multi_cell(h=5.0, w=0, txt=file_name)
            pdf.ln(5)
            file_features = get_tfidf_features(file_name)
            # print("Total features: {} for: {}".format(len(set(file_features)), file_name.split('/')[-1]))
            pdf.multi_cell(h=5.0, w=0, txt="Total features: {} for: {}".format(len(set(file_features)),
                                                                               file_name.split('/')[-1]))
            pdf.ln(5)
            common_features_per = float(len(set(file_features) & set(input_features)))*100/len(set(file_features))
            common_features[file_name] = common_features_per 
            # print("% of common features: {} for: {}".format(common_features_per, file_name))
            pdf.multi_cell(h=5.0, w=0, txt="% of common features: {} for: {}".format(common_features_per, file_name))
            pdf.ln(5)
    file_dict = {k: v for k, v in sorted(common_features.items(), key=lambda item: item[1])}
    # print(common_features)
    pdf.multi_cell(h=5.0, w=0, txt=common_features)
    pdf.ln(5)
    x=sorted(file_dict, key=file_dict.get, reverse=True)
    return sorted(file_dict, key=file_dict.get, reverse=True)[0]


def transform_features(vectorizer, df_info):
    df_info = df_info.astype(str)
    df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
    processed_text = text_processing(list(df_info['comb_text']))
    X = vectorizer.transform(processed_text)
    tfidf_matrix = X.todense()
    return tfidf_matrix


def print_steps_taken():
    pdf.ln(5)
    pdf.set_font('Arial', 'BU')
    pdf.multi_cell(h=7.5, w=0, txt="B. Steps Taken to Run the Predictive Models")
    pdf.set_font('Arial')
    pdf.ln(4)
    pdf.multi_cell(h=5.0, w=0, txt="1. Read the input data file provided.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="2. Data cleaning: remove null rows and columns and impute missing values.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="3. Identify columns containing categorical and textual data and convert it to "
                                   "numerical vectors.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="4. Assign target value: Target values are the dependent variable.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="5. Splitting the dataset for training and testing to train total of 10 different "
                                   "classifiers.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="6. Calculate Feature Importance for each classifier. Feature importance gives "
                                   "a score for each feature of your data.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="7. Plot Confusion Matrix and Classification report. A confusion matrix is a table "
                                   "that is used to describe the performance of a model.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="8. Identify and select top 5 classifiers using the F1-Score. The F1-score is a "
                                   "measure of a test's (model's) accuracy.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="9. Receiver Operating Characteristic (ROC) Curve. ROC is a probability curve. It "
                                   "tells how much model is capable of distinguishing between classes.")
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="10.  Identify optimal threshold (accuracy of the model) and predict.")
    pdf.ln(10)


if __name__ == "__main__":
    plots_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Plots"))
    files = glob.glob('{}/*.png'.format(plots_path))
    logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "logo"))
    pdf.image("{}/logo-new.png".format(logo_path), w=135, h=50, x=40)
    pdf.ln(5)
    today = date.today()
    today_date = today.strftime("%B %d, %Y")
    # pdf.multi_cell(h=5.0, w=0, txt="PREDICT ME", align="C")
    # pdf.ln(5)
    pdf.multi_cell(h=5.0, w=0, txt="Predictive Modeling Results", align="C")
    pdf.ln(3)
    pdf.multi_cell(h=5.0, w=0, txt="Report Date: {}".format(today_date), align="C")
    pdf.ln(5)
    for f in files:
        os.remove(f)
    file_path = sys.argv[1]
    donation_columns = ast.literal_eval(sys.argv[2])
    donor_df = read_input_file(file_path)
    # print("all columns: {}".format(donor_df.columns))
    # print(donor_df.shape)
    # print("donation columns: {}".format(donation_columns))
    pdf.set_font('Arial', 'BU')
    pdf.multi_cell(h=7.5, w=0, txt="A. Data Input Summary")
    pdf.set_font('Arial')
    pdf.ln(4)
    donor_df = remove_rows_containg_all_null_values(donor_df)
    pdf.multi_cell(h=5.0, w=0, txt="1. Total Data Sample: {}".format(donor_df.shape[0]))
    pdf.ln(2.5)
    pdf.multi_cell(h=5.0, w=0, txt="2. Donation Columns:")
    pdf.ln(2.5)
    test_list = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    for i in range(len(donation_columns)):
        pdf.multi_cell(h=5.0, w=0, txt="          {}. {}".format(test_list[i], donation_columns[i]))
        pdf.ln(2.5)
    # donation_columns = identify_years_columns(donor_df)
    donation_columns_df = process_donation_columns(donor_df, donation_columns)
    postive_class = donation_columns_df[donation_columns_df['target'] == 1].shape[0]
    negative_class = donation_columns_df[donation_columns_df['target'] == 0].shape[0]
    no_donations_columns = False
    if (len(donation_columns) == 0) | ((postive_class <= (donation_columns_df.shape[0])*0.02) |
                                       (negative_class <= (donation_columns_df.shape[0])*0.02)):
        new_file = find_similar_files(file_path)
        # print(new_file)
        pdf.multi_cell(h=5.0, w=0, txt=new_file)
        pdf.ln(5)
        df = read_input_file(new_file)
        df = remove_rows_containg_all_null_values(df)
        donation_columns = identify_years_columns(df)
        donation_columns_df = process_donation_columns(df, donation_columns)
        no_donations_columns = True
    else:
        df = donor_df

    info_columns = identify_info_columns(df, donation_columns, True)
    df_info = remove_columns_unique_values(df, info_columns)
    # print(df_info.shape)
    # print(df_info.columns)
    cat_col = identify_categorical_columns(df, info_columns)
    # print("Categorical columns: {}".format(cat_col))

    pdf.multi_cell(h=5.0, w=0, txt="3. Categorical Columns:")
    pdf.ln(3)
    test_list = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    for i in range(len(cat_col)):
        pdf.multi_cell(h=5.0, w=0, txt="          {}. {}".format(test_list[i], cat_col[i]))
        pdf.ln(2.5)

    processed_text, tfidf_matrix, feature_names, df_info, vectorizer = feature_extraction(df_info)
    y = list(donation_columns_df['target'])

    if no_donations_columns:
        X_pred = transform_features(vectorizer, donor_df)
        X_train = tfidf_matrix
    else:
        X_pred = tfidf_matrix
        X_train = tfidf_matrix
        del donation_columns_df['target']
        # column_train = sorted(donation_columns_df.columns)[:-1]
        # column_predict = sorted(donation_columns_df.columns)[1:]
        # print(column_train, column_predict)
        # column_train = donation_columns_df[column_train].values.tolist()
        # column_predict = donation_columns_df[column_predict].values.tolist()
        # print(tfidf_matrix.shape, X_pred.shape)
        # X_train = np.append(tfidf_matrix, column_train, 1)
        # X_pred = np.append(tfidf_matrix, column_predict, 1)
        # print(X_train.shape, X_pred.shape)

    model_f1_score, classification_full_pred, classification_full_pred_prob, feature_importance_dict, roc_fpr, \
    roc_tpr, roc_auc, y_test_dict, y_pred_dict = model_selection(X_train, y, X_pred)
    df_final = generate_prediction_file(donor_df, model_f1_score, classification_full_pred,
                                        classification_full_pred_prob, y, feature_importance_dict, roc_fpr, roc_tpr,
                                        roc_auc, y_test_dict, y_pred_dict, feature_names, df_info, donation_columns_df)

    prediction_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediction"))
    pdf_report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "PdfReport"))
    file_path = file_path.split("/")[-1]
    pdf.output("{}/{}_report.pdf".format(pdf_report_path, file_path.split(".")[0]))
    df_final.to_csv("{}/{}_prediction.csv".format(prediction_path, file_path.split(".")[0]), index=None)

