import LinReg
import main
import numpy as np
import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt
import pickle

#UTIL
linreg = LinReg.LinReg(x_train, y_train)
x_train, x_reg, y_train, y_reg = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2)
pickle.dump(linreg, open('simples.pkl', 'wb'), protocol=4)
plt.plot(range(len(linreg.js)), linreg.js)
plt.show()

#NORMAL
x_train, y_train = main.extrair(main.muda_col(main.tratar('train.csv')))
x_train = x_train.values
y_train = y_train.values
x_test = main.tratar('test.csv').values
y_test = pd.read_csv('test_target.csv')['shares'].values

x_train, x_reg, y_train, y_reg = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2)

# COM QUADRADOS
x_train, y_train = main.extrair(main.muda_col(main.tratar('train.csv')))
x_train = main.eleva(x_train, 2).values
y_train = y_train.values
x_test = main.eleva(main.tratar('test.csv'), 2).values
y_test = pd.read_csv('test_target.csv')['shares'].values
x_train, x_reg, y_train, y_reg = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2)

#COM CUBOS
x_train, y_train = main.extrair(main.muda_col(main.tratar('train.csv')))
x_train_inter = main.eleva(x_train, 2)
dfn = x_train
dfn = dfn.add_suffix('_3')
dfn.apply(lambda x: x**3)
x_train = pd.concat([x_train_inter, dfn], axis=1).values
y_train = y_train.values
x_test = main.tratar('test.csv')
x_test_inter = main.eleva(x_test, 2)
dfn = x_test
dfn = dfn.add_suffix('_3')
dfn.apply(lambda x: x**3)
x_test = pd.concat([x_test_inter, dfn], axis=1).values
y_test = pd.read_csv('test_target.csv')['shares'].values

#TOP10
x_train, y_train = main.extrair(main.muda_col(main.tratar('train.csv')))
x_train_top = x_train.iloc[:, [0,12,13,14,15,16,17,20,23,25,26,27,29,31,32,34]].values
x_test = main.tratar('test.csv')
x_test_top = x_test.iloc[:, [0,12,13,14,15,16,17,20,23,25,26,27,29,31,32,34]].values
y_train = y_train.values
y_test = pd.read_csv('test_target.csv')['shares'].values
x_train_top, x_reg_top, y_train, y_reg = sklearn.model_selection.train_test_split(x_train_top, y_train, test_size=0.2)

#TOP10 COM QUADRADOS  E MUDANCAS
x_train, y_train = main.extrair(main.muda_col(main.tratar('train.csv')))
x_train_top = x_train.iloc[:, [0,12,13,14,15,16,17,20,23,25,26,27,29,31,32,34]]
x_test = main.tratar('test.csv')
x_test_top = x_test.iloc[:, [0,12,13,14,15,16,17,20,23,25,26,27,29,31,32,34]]
y_train = y_train.values
y_test = pd.read_csv('test_target.csv')['shares'].values
square_list=[1,2,3,4,5,6]
ln_list=[7,9,11,12,10]
for i in ln_list:
	x_test_top.iloc[:,i] = x_test_top.iloc[:,i].apply(lambda x: np.log(x))
for i in square_list:
	x_test_top['square' + str(i)] = x_test_top.iloc[:,i].apply(lambda x: x**2)


#SEM VARIAVEIS DISCRETAS
drop_list = ['day0', 'day1', 'day2', 'data_channel_is_lifestyle', 'data_channel_is_entertainment']
drop_list += ['data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']
x_train, y_train = main.extrair(main.muda_col(main.tratar('train.csv')))
x_train_drop = x_train.drop(drop_list, axis=1).values
y_train = y_train.values
x_test = main.tratar('test.csv')
x_test_drop = x_test.drop(drop_list, axis=1).values
y_test = pd.read_csv('test_target.csv')['shares'].values
x_train_drop, x_reg_drop, y_train, y_reg = sklearn.model_selection.train_test_split(x_train_drop, y_train, test_size=0.2)
