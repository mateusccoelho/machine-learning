import LinReg
import numpy as np
import pandas as pd
#import sklearn

def eleva(df, n):
	dfn = df
	dfn = dfn.add_suffix('_' + str(n))
	dfn.apply(lambda x: x**n)
	dfn = pd.concat([df, dfn], axis=1)
	return dfn

def tratar(nome_arq):
	# Le entrada
	test_data = pd.read_csv(nome_arq)

	# Transforma coluna url em x0 correspondente a coef0
	test_data.rename(columns={'url':'x0'}, inplace=True)
	test_data['x0'] = 1.0

	# Remove timedelta (non-predictive) e is_weekend (repetido)
	test_data.drop(['timedelta', 'is_weekend'], axis=1, inplace=True)

	# Adicionando colunas dos dias
	test_data['day0'] = 0.0
	test_data['day1'] = 0.0
	test_data['day2'] = 0.0

	# Mudando os valores dos dias
	test_data.loc[test_data.weekday_is_monday == 1.0, ['day0', 'day1', 'day2']] = 0.0,0.0,0.0
	test_data.loc[test_data.weekday_is_tuesday == 1.0, ['day0', 'day1', 'day2']] = 0.0,0.0,1.0
	test_data.loc[test_data.weekday_is_wednesday == 1.0, ['day0', 'day1', 'day2']] = 0.0,1.0,0.0
	test_data.loc[test_data.weekday_is_thursday == 1.0, ['day0', 'day1', 'day2']] = 0.0,1.0,1.0
	test_data.loc[test_data.weekday_is_friday == 1.0, ['day0', 'day1', 'day2']] = 1.0,0.0,0.0
	test_data.loc[test_data.weekday_is_saturday == 1.0, ['day0', 'day1', 'day2']] = 1.0,0.0,1.0
	test_data.loc[test_data.weekday_is_sunday == 1.0, ['day0', 'day1', 'day2']] = 1.0,1.0,0.0

	# Deletando as colunas de dias
	dias = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday']
	dias += ['weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
	test_data.drop(dias, axis=1, inplace=True)

	# Normalizando features
	# columns = [1,2,6,7,8,9,10,11,18,19,20,21,22,23,24,25,26,27,28,29] # colunas pra normalizar
	# for i in columns:
	# 	media = test_data.iloc[:,i].mean()
	# 	max = test_data.iloc[:,i].max()
	# 	min = test_data.iloc[:,i].min()
	# 	test_data[test_data.columns[i]] = test_data.iloc[:,i].apply(lambda x: (x - media)/(max - min))

	return test_data

def muda_col(test_data):
	# Reordenador colunas
	cols = test_data.columns.tolist()
	del cols[51]
	cols.append('shares')
	test_data = test_data[cols]
	return test_data

def extrair(test_data):
	# Partir os dados
	x = test_data.iloc[:,:-1]
	y = test_data.iloc[:,-1]
	return x,y
