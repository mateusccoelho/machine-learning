import numpy as np

class LinReg:

	def __init__(self, x, y):
		self.x =  x
		self.y = y
		self.m = self.x.shape[0] # numero de instancias
		self.n = self.x.shape[1] # numero de features
		self.lbd = 0
		self.alfa = 0.0001
		self.coef = None
		self.js = []

	# calcula coeficientes pelas equacoes normais
	def normal_eq(self):
		id = np.identity(self.n) # cria id matrix de n features
		id[0,0] = 0
		id = id * self.lbd
		xt = self.x.transpose()
		self.coef = np.linalg.pinv(xt @ self.x + id) @ xt @ self.y

	# Retorna j e erro
	def j(self, x_qlq, y_qlq):
		reg_sum = (self.coef * self.coef).sum() * self.lbd
		error = x_qlq @ self.coef - y_qlq
		error_sum = (error * error).sum()
		j = (error_sum + reg_sum) / (2*self.m)
		return j, error

	# retorna coast function j e MAE
	def score(self, x_test, y_test):
		j_value, error = self.j(x_test, y_test)
		mae = np.absolute(error).mean()
		return j_value, mae

	def gd(self):
		self.coef = np.zeros((self.n,))
		self.js = []
		j_ant = np.inf
		j_at, error = self.j(self.x, self.y)
		self.js.append(j_at)
		const = 1 - self.alfa * (self.lbd / self.m)
		novo_coef = np.empty(shape=(self.n,))
		while(j_ant - j_at > 10):
			if(len(self.js) < 50000):
				for i in range(self.n):
					somatorio = (error * self.x[:,i]).sum()
					novo_coef[i] = self.coef[i] * const - (self.alfa / self.m) * somatorio
				self.coef = novo_coef
				j_ant = j_at
				j_at, error = self.j(self.x, self.y)
				self.js.append(j_at)
				print(len(self.js))
			else:
				break
