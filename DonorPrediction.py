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


import sys
import warnings
import time
import os
import ast
import glob

warnings.filterwarnings("ignore")
pp = PdfPages('plots.pdf')


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
		print("{} file format is not supported".format(extension))


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
	print("Total training samples: {}".format(number_of_sample))
	for col in column_names:
		print("Column: {}, Null count: {}".format(col, df[col].isnull().sum()))
		if df[col].isnull().sum() <= number_of_sample/2:
			final_col_.append(col)
	final_col = []
	print("_____________\n")
	for col in final_col_:
		print("Column: {}, Unique count: {}".format(col, df[col].unique().shape[0]))
		if (df[col].unique().shape[0] <= number_of_sample*0.8) and (df[col].unique().shape[0]!=1) :
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
	print("Number of features used: {}".format(feature_count))
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
	print(donation_columns['target'].value_counts())
	return donation_columns


def generate_correlation(donation_columns):
	fig, ax = plt.subplots(figsize=(8, 8))
	ax = sn.heatmap(donation_columns.corr(), annot=True)


def get_feature_weights(feature_list, feature_dict):
	sum_ = 0
	for f in feature_list:
		sum_ += feature_dict.get(f.lower(),0)
	return sum_


def calculate_feature_importance(df_info, feature_dict):
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

	# featfig = plt.figure(figsize=(10,6))
	# featax = featfig.add_subplot(1, 1, 1)
	# featax.barh(pos, sorted(feature_imp), align='center')
	# featax.set_yticks(pos)
	# featax.set_yticklabels(np.array(feature_columns)[sorted_idx], fontsize=8)
	# featax.set_xlabel('Relative Feature Importance')

	# plt.tight_layout()
	# pp.savefig(plt)
	# plt.show()


def print_confusion_matrix_classification_report(y_test, y_pred):
	# df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2), range(2))
	# sn.set(font_scale=1.4) # for label size
	# sn.heatmap(df_cm, annot=True,fmt="d", annot_kws={"size": 16}) # font size
	# plt.xlabel("Predicted")
	# plt.ylabel("True")
	# pp.savefig(sn)
	# plt.show()
	
	print("classification report")
	print(classification_report(y_test, y_pred))
	print("___________________________\n")


def model_selection(X, y, X_pred, feature_names, df_info):
	models = [{'label': 'LogisticRegression', 'model': LogisticRegression()},
				{'label': 'GaussianNB', 'model': GaussianNB()},
				{'label': 'MultinomialNB', 'model': MultinomialNB()},
				{'label': 'ComplementNB', 'model': ComplementNB()},
				{'label': 'BernoulliNB', 'model': BernoulliNB()},
				{'label': 'DecisionTreeClassifier', 'model': DecisionTreeClassifier()},
				{'label': 'SGDClassifier', 'model': SGDClassifier(loss='log')},
				{'label': 'PassiveAggressiveClassifier', 'model': PassiveAggressiveClassifier()}, #No predict_proba
				{'label': 'LinearSVC', 'model': LinearSVC()}, #No predict_proba
				{'label': 'RandomForestClassifier', 'model': RandomForestClassifier()}]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	plt.figure(figsize=(15,10))
	model_f1_score={}
	classification_full_pred={}
	classification_full_pred_prob={}
	for m in models:
		start_time = time.time()
		model = m['model']
		if m['label'] in ['PassiveAggressiveClassifier', 'LinearSVC']:
			model = CalibratedClassifierCV(model)
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		classification_full_pred[m['label']] = model.predict(X_pred) ## ADD New columns
		classification_full_pred_prob[m['label']] = model.predict_proba(X_pred)
		print("Classifier: {} and time(seconds): {}".format(m['label'], round(time.time()-start_time, 3)))
		print()
		print("Classifier: {} and f1-score {}".format(m['label'], round(f1_score(y_test, y_pred, average='weighted'), 2)))
		model_f1_score[m['label']] = round(f1_score(y_test, y_pred, average='weighted'), 2)
		print_confusion_matrix_classification_report(y_test, y_pred)
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

		feature_dict = {i: abs(j) for i, j in zip(feature_names, feature_value)}
		calculate_feature_importance(df_info, feature_dict)
		try:
			fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
			auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
		except:
			fpr, tpr, thresholds = roc_curve(y_test, y_pred)
			auc = roc_auc_score(y_test, y_pred)
		# plt.plot(fpr, tpr, label="{} ROC (area = {})".format(m['label'], round(auc,2)))
		# pp.savefig(plt)
	
	# plt.plot([0, 1], [0, 1],'r--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('1-Specificity(False Positive Rate)')
	# plt.ylabel('Sensitivity(True Positive Rate)')
	# plt.title('Receiver Operating Characteristic')
	# plt.legend(loc="lower right")
	# pp.savefig(plt)
	# plt.show()
	return model_f1_score, classification_full_pred, classification_full_pred_prob


def generate_prediction_file(df, model_f1_score, classification_full_pred, classification_full_pred_prob):
	model_f1_score = {k: v for k, v in sorted(model_f1_score.items(), key=lambda item: item[1])}
	top_5_model = sorted(model_f1_score, key=model_f1_score.get, reverse=True)[:5]
	# print(top_5_model, model_f1_score, classification_full_pred.keys(), classification_full_pred_prob.keys())
	for m in top_5_model:
		prediction = classification_full_pred.get(m)
		prob = classification_full_pred_prob.get(m)
		df['2020_{}'.format(m)] = prediction
		df['donor_prob_{}'.format(m)] = [round(prob[x][1], 2) for x in range(len(prob))]
		df['non_donor_prob_{}'.format(m)] = [round(prob[x][0], 2) for x in range(len(prob))]
		print(df['2020_{}'.format(m)].value_counts())
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
			print(file_name, input_file)
			file_features = get_tfidf_features(file_name)
			print("Total features: {} for: {}".format(len(set(file_features)), file_name.split('/')[-1]))
			common_features_per = float(len(set(file_features) & set(input_features)))*100/len(set(file_features))
			common_features[file_name] = common_features_per 
			print("% of common features: {} for: {}".format(common_features_per, file_name))
	file_dict = {k: v for k, v in sorted(common_features.items(), key=lambda item: item[1])}
	print(common_features)
	x=sorted(file_dict, key=file_dict.get, reverse=True)
	return sorted(file_dict, key=file_dict.get, reverse=True)[0]


def transform_features(vectorizer, df_info):
	df_info = df_info.astype(str)
	df_info['comb_text'] = df_info.apply(lambda x: ' '.join(x), axis=1)
	processed_text = text_processing(list(df_info['comb_text']))
	X = vectorizer.transform(processed_text)
	tfidf_matrix = X.todense()
	return tfidf_matrix


if __name__ == "__main__":
	file_path = sys.argv[1]
	donation_columns = ast.literal_eval(sys.argv[2])
	donor_df = read_input_file(file_path)
	print("all columns: {}".format(donor_df.columns))
	print(donor_df.shape)
	print("donation columns: {}".format(donation_columns))
	donor_df = remove_rows_containg_all_null_values(donor_df)
	# donation_columns = identify_years_columns(donor_df)
	no_donations_columns = False
	if len(donation_columns) == 0:
		new_file = find_similar_files(file_path)
		print(new_file)
		df = read_input_file(new_file)
		df = remove_rows_containg_all_null_values(df)
		donation_columns = identify_years_columns(df)
		no_donations_columns = True
	else:
		df = donor_df
	donation_columns_df = process_donation_columns(df, donation_columns)
	postive_class = donation_columns_df[donation_columns_df['target'] == 1].shape[0]
	negative_class = donation_columns_df[donation_columns_df['target'] == 0].shape[0]
	if (postive_class <= (donation_columns_df.shape[0])*0.02) | (negative_class <= (donation_columns_df.shape[0])*0.02):
		print("Data is skewed")
		print("Postive class count: {} and negative class count: {}".format(postive_class, negative_class))
	else:
		info_columns = identify_info_columns(df, donation_columns, True)
		print(len(info_columns))
		df_info = remove_columns_unique_values(df, info_columns)
		print(df_info.shape)
		print(df_info.columns)
		cat_col = identify_categorical_columns(df, info_columns)
		print("Categorical columns: {}".format(cat_col))
		processed_text, tfidf_matrix, feature_names, df_info, vectorizer = feature_extraction(df_info)
		y = list(donation_columns_df['target'])
		# generate_correlation(donation_columns_df)
		if no_donations_columns:
			X_pred = transform_features(vectorizer, donor_df)
			X_train = tfidf_matrix
		else:
			X_pred = tfidf_matrix
			del donation_columns_df['target']
			column_train = sorted(donation_columns_df.columns)[:-1]
			column_predict = sorted(donation_columns_df.columns)[1:]
			# print(column_train, column_predict)
			column_train = donation_columns_df[column_train].values.tolist()
			column_predict = donation_columns_df[column_predict].values.tolist()
			# print(tfidf_matrix.shape, X_pred.shape)
			X_train = np.append(tfidf_matrix, column_train, 1)
			X_pred = np.append(tfidf_matrix, column_predict, 1)
			# print(X_train.shape, X_pred.shape)

		model_f1_score, classification_full_pred, classification_full_pred_prob = model_selection(X_train, y, X_pred,
																								  feature_names,
																								  df_info)
		df_final = generate_prediction_file(donor_df, model_f1_score, classification_full_pred, classification_full_pred_prob)
		prediction_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediction"))
		file_path = file_path.split("/")[-1]
		df_final.to_csv("{}/{}_prediction.csv".format(prediction_path, file_path.split(".")[0]), index=None)
